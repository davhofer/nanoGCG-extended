import copy
import gc
import logging
import os

from dataclasses import dataclass, field
from enum import Enum
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor
from torch.profiler import profile, ProfilerActivity, schedule
from transformers import set_seed, DynamicCache

from nanogcg.utils import (
    INIT_CHARS,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_trace_handler(output_dir, filename_prefix):
    def trace_handler(prof):
        try:
            trace_file = f"{output_dir}/{filename_prefix}_{prof.step_num}.json"
            prof.export_chrome_trace(trace_file)
            logger.info(f"Profiling trace saved to: {trace_file}")
        except Exception as e:
            logger.error(f"Failed to save profiling trace: {e}")

    return trace_handler


class ESMetric(Enum):
    MATCH = 1
    LOSS = 2


class ESCondition(Enum):
    ALL = 1
    ANY = 2
    FRACTION = 3


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    metric: ESMetric = ESMetric.MATCH  # Stop early if output string (argmax of logits) is a perfect match or if loss is low
    condition: ESCondition = (
        ESCondition.ALL
    )  # Stop early if all, any, or a fraction of samples fulfill the criterion
    fraction: float = 0.5  # used with ESCondition.FRACTION. fraction of samples that needs to fulfill the criterion to stop early
    loss_threshold: float = 0.001


@dataclass
class ProfilingConfig:
    enabled: bool = False
    wait: int = 2
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    record_shapes: bool = False
    profile_memory: bool = True
    with_stack: bool = True
    output_dir: str = "./profiling_results"
    filename_prefix: str = "trace"


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    use_autoregressive_loss: bool = True
    use_autoregressive_gradients: bool = False
    use_straight_through_estimator: bool = (
        False  # Use STE for differentiable generation
    )
    seed: int = None
    verbosity: str = "INFO"
    debug: bool = False
    # Profiling options
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    match_fraction_at_best: float  # Fraction of exact matches when best string was found


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer: list[
            tuple[float, Tensor]
        ] = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(
        torch.rand((search_width, n_optim_tokens), device=grad.device)
    )[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )

    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = (
            None
            if config.allow_non_ascii
            else get_nonascii_toks(tokenizer, device=model.device)
        )
        self.prefix_cache = None

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )

        if model.device == torch.device("cpu"):
            logger.warning(
                "Model is on the CPU. Use a hardware accelerator for faster optimization."
            )

        if not tokenizer.chat_template:
            logger.warning(
                "Tokenizer does not have a chat template. Assuming base model and setting chat template to empty."
            )
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            )

    def _setup_profiler(self, config: ProfilingConfig):
        """Initialize and return the torch profiler for Chrome/Perfetto trace output."""
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)

        # Determine activities based on device
        activities = [ProfilerActivity.CPU]
        if self.model.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        logger.info(
            f"Profiling enabled. Chrome traces will be saved to {config.output_dir}"
        )
        logger.info("View traces at: chrome://tracing/ or https://ui.perfetto.dev/")

        return profile(
            activities=activities,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
            schedule=schedule(
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=config.repeat,
            ),
            on_trace_ready=get_trace_handler(config.output_dir, config.filename_prefix),
        )

    def _optimization_loop(
        self, config, buffer, optim_ids, losses, optim_strings, tokenizer, prof=None
    ):
        """Run the main GCG optimization loop."""
        # Track match fractions for each step
        match_fractions = []
        for step in tqdm(range(config.num_steps)):
            # Compute the token gradient across all prompts
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]

                # Compute loss on all candidate sequences across all prompts
                batch_size = (
                    new_search_width if config.batch_size is None else config.batch_size
                )

                # Choose evaluation method based on config
                if config.use_autoregressive_loss:
                    loss = find_executable_batch_size(
                        self._compute_candidates_loss_autoregressive, batch_size
                    )(sampled_ids)
                else:
                    loss = find_executable_batch_size(
                        self._compute_candidates_loss, batch_size
                    )(sampled_ids)
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            
            # Calculate match fraction for current best string
            current_match_fraction = self._calculate_match_fraction(optim_ids)
            match_fractions.append(current_match_fraction)

            buffer.log_buffer(tokenizer)

            # profiler step
            if prof is not None:
                prof.step()

            if self.stop_flag:
                logger.info(
                    f"Early stopping triggered  (Config: {self.config.early_stop})"
                )
                break
        
        # Return match fractions for use in result creation
        return match_fractions

    def run(
        self,
        messages_and_targets: List[Tuple[Union[str, List[dict]], str]],
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Prepare all prompts and targets
        self.prompt_data = []
        for messages, target in messages_and_targets:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            else:
                messages = copy.deepcopy(messages)

            # Append the GCG string at the end of the prompt if location not specified
            if not any(["{optim_str}" in d["content"] for d in messages]):
                messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

            template = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")
            # Handle multiple {optim_str} occurrences by using only the first one
            parts = template.split("{optim_str}")
            if len(parts) < 2:
                raise ValueError("No {optim_str} placeholder found in template")
            elif len(parts) == 2:
                before_str, after_str = parts
            else:
                # Multiple {optim_str} - use first occurrence and rejoin the rest
                before_str = parts[0]
                after_str = "".join(parts[1:])
                print(
                    "Warning: Multiple {optim_str} placeholders found, using first occurrence and removing the rest."
                )

            target = " " + target if config.add_space_before_target else target

            # Tokenize everything that doesn't get optimized
            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device, torch.int64)
            after_ids = tokenizer(
                [after_str], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)
            target_ids = tokenizer(
                [target], add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device, torch.int64)

            # Embed everything that doesn't get optimized
            embedding_layer = self.embedding_layer
            before_embeds, after_embeds, target_embeds = [
                embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
            ]

            prompt_data = {
                "before_embeds": before_embeds,
                "after_embeds": after_embeds,
                "target_embeds": target_embeds,
                "target_ids": target_ids,
                "before_str": before_str,
                "after_str": after_str,
                "target": target,
            }

            # Compute prefix cache for this prompt
            if config.use_prefix_cache:
                with torch.no_grad():
                    output = model(inputs_embeds=before_embeds, use_cache=True)
                    # TODO: why use DynamicCache.from_legacy_cache here?
                    prompt_data["prefix_cache"] = DynamicCache.from_legacy_cache(
                        output.past_key_values
                    )

            self.prompt_data.append(prompt_data)

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        # Run optimization loop with or without profiling
        if config.profiling.enabled:
            prof = self._setup_profiler(config.profiling)
            prof.start()
            try:
                match_fractions = self._optimization_loop(
                    config,
                    buffer,
                    optim_ids,
                    losses,
                    optim_strings,
                    tokenizer,
                    prof=prof,
                )
            finally:
                prof.stop()
        else:
            match_fractions = self._optimization_loop(
                config, buffer, optim_ids, losses, optim_strings, tokenizer
            )

        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            match_fraction_at_best=match_fractions[min_loss_index] if match_fractions else 0.0,
        )

        return result

    def init_buffer(self) -> AttackBuffer:
        """Initialize attack buffer for multi-prompt optimization."""
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(
            f"Initializing multi-prompt attack buffer of size {config.buffer_size}..."
        )

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(
                config.optim_str_init, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = (
                    tokenizer(
                        INIT_CHARS, add_special_tokens=False, return_tensors="pt"
                    )["input_ids"]
                    .squeeze()
                    .to(model.device)
                )
                init_indices = torch.randint(
                    0,
                    init_buffer_ids.shape[0],
                    (config.buffer_size - 1, init_optim_ids.shape[1]),
                )
                init_buffer_ids = torch.cat(
                    [init_optim_ids, init_buffer_ids[init_indices]], dim=0
                )
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(
                    f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}"
                )
            try:
                init_buffer_ids = tokenizer(
                    config.optim_str_init, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(model.device)
            except ValueError:
                logger.error(
                    "Unable to create buffer. Ensure that all initializations tokenize to the same length."
                )

        true_buffer_size = max(1, config.buffer_size)

        # Compute the loss on the initial buffer entries across all prompts
        # Use the same evaluation method as configured for the main optimization loop
        if config.use_autoregressive_loss:
            init_buffer_losses = find_executable_batch_size(
                self._compute_candidates_loss_autoregressive, true_buffer_size
            )(init_buffer_ids)
        else:
            init_buffer_losses = find_executable_batch_size(
                self._compute_candidates_loss, true_buffer_size
            )(init_buffer_ids)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized multi-prompt attack buffer.")

        return buffer

    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix across all prompts.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        if self.config.use_autoregressive_gradients:
            return self.compute_token_gradient_autoregressive(optim_ids)
        else:
            return self.compute_token_gradient_teacher_forcing(optim_ids)

    def compute_token_gradient_teacher_forcing(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient using teacher forcing (original method).

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        total_loss = 0.0
        for prompt_data in self.prompt_data:
            if prompt_data.get("prefix_cache"):
                input_embeds = torch.cat(
                    [
                        optim_embeds,
                        prompt_data["after_embeds"],
                        prompt_data["target_embeds"],
                    ],
                    dim=1,
                )
                output = model(
                    inputs_embeds=input_embeds,
                    past_key_values=prompt_data["prefix_cache"],
                    use_cache=True,
                )
            else:
                input_embeds = torch.cat(
                    [
                        prompt_data["before_embeds"],
                        optim_embeds,
                        prompt_data["after_embeds"],
                        prompt_data["target_embeds"],
                    ],
                    dim=1,
                )
                output = model(inputs_embeds=input_embeds)

            logits = output.logits

            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - prompt_data["target_ids"].shape[1]
            shift_logits = logits[
                ..., shift - 1 : -1, :
            ].contiguous()  # (1, num_target_ids, vocab_size)
            shift_labels = prompt_data["target_ids"]

            if self.config.use_mellowmax:
                label_logits = torch.gather(
                    shift_logits, -1, shift_labels.unsqueeze(-1)
                ).squeeze(-1)
                loss = mellowmax(
                    -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                )
            else:
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

            total_loss += loss

        # Average the loss across all prompts
        avg_loss = total_loss / len(self.prompt_data)

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[avg_loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def compute_token_gradient_autoregressive(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient using autoregressive generation to match the autoregressive loss.

        This method generates tokens autoregressively (without teacher forcing) and computes
        gradients at each step, providing a better match to the inference-time behavior.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(
            optim_ids, num_classes=embedding_layer.num_embeddings
        )
        optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        total_loss = 0.0
        for prompt_data in self.prompt_data:
            # Get target information
            target_ids = prompt_data["target_ids"]
            target_length = target_ids.shape[1]

            # Initialize loss accumulator for this prompt
            sequence_loss = 0.0

            # Prepare initial embeddings and cache for autoregressive generation
            if prompt_data.get("prefix_cache"):
                # Start with optim + after embeddings (prefix is already cached)
                current_embeds = torch.cat(
                    [
                        optim_embeds,
                        prompt_data["after_embeds"],
                    ],
                    dim=1,
                )
                current_cache = prompt_data["prefix_cache"]
            else:
                # Without prefix cache, start with full prompt embeddings
                current_embeds = torch.cat(
                    [
                        prompt_data["before_embeds"],
                        optim_embeds,
                        prompt_data["after_embeds"],
                    ],
                    dim=1,
                )
                current_cache = None

            # Generate tokens autoregressively and compute loss at each step
            for t in range(target_length):
                # Forward pass with current embeddings
                output = model(
                    inputs_embeds=current_embeds,
                    past_key_values=current_cache,
                    use_cache=True,
                )

                # Get logits for the last position (next token prediction)
                logits = output.logits[:, -1, :]

                # Compute log probabilities
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Get the log probability of the target token at this position
                target_token = target_ids[:, t]
                token_log_prob = log_probs.gather(1, target_token.unsqueeze(1)).squeeze(
                    1
                )

                # Accumulate negative log likelihood (loss)
                step_loss = -token_log_prob
                sequence_loss += step_loss

                # For next step: use the predicted token (greedy decoding)
                # This is the key difference from teacher forcing
                if self.config.use_straight_through_estimator:
                    # Straight-Through Estimator: use hard argmax in forward pass,
                    # but allow gradients to flow through soft probabilities in backward pass
                    next_token_hard = logits.argmax(dim=-1)
                    next_token_soft = torch.nn.functional.softmax(logits, dim=-1)

                    # Create one-hot encoding of hard decision
                    next_token_onehot = torch.nn.functional.one_hot(
                        next_token_hard, num_classes=embedding_layer.num_embeddings
                    ).to(logits.dtype)

                    # STE trick: forward uses hard, backward uses soft
                    next_token_ste = (
                        next_token_onehot - next_token_soft.detach() + next_token_soft
                    )

                    # Compute embeddings using the STE tokens
                    next_embeds = next_token_ste @ embedding_layer.weight
                    next_embeds = next_embeds.unsqueeze(1)
                else:
                    # Original implementation: detached argmax
                    with torch.no_grad():
                        next_token = logits.argmax(dim=-1)
                        next_embeds = embedding_layer(next_token).unsqueeze(1)

                # Update inputs for next step
                current_embeds = next_embeds
                current_cache = output.past_key_values

            # Average loss over sequence length to get per-token loss
            prompt_loss = sequence_loss / target_length
            total_loss += prompt_loss

        # Average the loss across all prompts
        avg_loss = total_loss / len(self.prompt_data)

        optim_ids_onehot_grad = torch.autograd.grad(
            outputs=[avg_loss], inputs=[optim_ids_onehot]
        )[0]

        return optim_ids_onehot_grad

    def _check_early_stopping(
        self,
        prompt_data: dict,
        current_batch_size: int,
        shift_logits: torch.Tensor = None,
        shift_labels: torch.Tensor = None,
        loss: torch.Tensor = None,
        generated_tokens: torch.Tensor = None,
    ) -> bool:
        """Check early stopping conditions and log debug messages.

        Args:
            prompt_data: Dictionary containing prompt information
            current_batch_size: Size of current batch
            shift_logits: Logits for teacher forcing evaluation (optional)
            shift_labels: Target labels for teacher forcing evaluation (optional)
            loss: Computed loss values (optional, for loss-based stopping)
            generated_tokens: Generated token sequences (optional, for autoregressive evaluation)

        Returns:
            bool: True if early stopping should be triggered
        """
        if not self.config.early_stop.enabled:
            return False

        should_stop = False

        if self.config.early_stop.metric == ESMetric.MATCH:
            if shift_logits is not None and shift_labels is not None:
                # Teacher forcing mode: check argmax matches
                matches = torch.all(
                    torch.argmax(shift_logits, dim=-1) == shift_labels,
                    dim=-1,
                )

                if self.config.debug and torch.any(matches):
                    matching_indices = torch.where(matches)[0]
                    for idx in matching_indices:
                        generated_str = self.tokenizer.decode(
                            torch.argmax(shift_logits[idx], dim=-1)
                        )
                        logger.debug(
                            f"Target string generated! Batch index {idx}: "
                            f"'{generated_str}' matches target '{prompt_data['target']}'"
                        )

            elif generated_tokens is not None:
                # Autoregressive mode: check exact sequence matches
                target_ids = prompt_data["target_ids"]
                expanded_target = target_ids.expand(current_batch_size, -1)
                matches = torch.all(generated_tokens == expanded_target, dim=-1)

                if self.config.debug and torch.any(matches):
                    matching_indices = torch.where(matches)[0]
                    for idx in matching_indices:
                        generated_str = self.tokenizer.decode(generated_tokens[idx])
                        logger.debug(
                            f"Exact match in autoregressive generation! Batch index {idx}: "
                            f"'{generated_str}' matches target '{prompt_data['target']}'"
                        )
            else:
                return False  # No valid inputs for match checking

            # Check stopping condition
            if self.config.early_stop.condition == ESCondition.ANY:
                should_stop = torch.any(matches).item()
            elif self.config.early_stop.condition == ESCondition.ALL:
                should_stop = torch.all(matches).item()
            elif self.config.early_stop.condition == ESCondition.FRACTION:
                fraction_matched = matches.float().mean().item()
                should_stop = fraction_matched >= self.config.early_stop.fraction

        elif self.config.early_stop.metric == ESMetric.LOSS:
            if loss is None:
                return False  # No loss provided for loss-based stopping

            # Check if loss is below threshold
            low_loss_mask = loss < self.config.loss_threshold

            if self.config.debug and torch.any(low_loss_mask):
                low_loss_indices = torch.where(low_loss_mask)[0]
                for idx in low_loss_indices:
                    logger.debug(
                        f"Very low loss achieved! Batch index {idx}: "
                        f"loss = {loss[idx].item():.4f} for target '{prompt_data['target']}'"
                    )

            # Check stopping condition
            if self.config.early_stop.condition == ESCondition.ANY:
                should_stop = torch.any(low_loss_mask).item()
            elif self.config.early_stop.condition == ESCondition.ALL:
                should_stop = torch.all(low_loss_mask).item()
            elif self.config.early_stop.condition == ESCondition.FRACTION:
                fraction_low_loss = low_loss_mask.float().mean().item()
                should_stop = fraction_low_loss >= self.config.early_stop.fraction

        return should_stop
    
    def _calculate_match_fraction(self, optim_ids: Tensor) -> float:
        """Calculate the fraction of prompts that produce exact target matches with current optim_ids."""
        match_count = 0
        total_prompts = len(self.prompt_data)
        
        with torch.no_grad():
            for prompt_data in self.prompt_data:
                # Create input embeddings with the current adversarial string
                input_embeds = torch.cat(
                    [
                        self.embedding_layer(prompt_data["input_ids"]),
                        self.embedding_layer(optim_ids),
                        self.embedding_layer(prompt_data["target_slice"]),
                    ],
                    dim=1,
                )
                
                # Forward pass to get logits
                outputs = self.model(inputs_embeds=input_embeds)
                logits = outputs.logits
                
                # Get the part of logits corresponding to the target
                shift_logits = logits[..., prompt_data["target_slice"].shape[1] - 1 : -1, :].contiguous()
                shift_labels = prompt_data["target_ids"].contiguous()
                
                # Check if argmax matches target exactly
                predicted_tokens = torch.argmax(shift_logits, dim=-1)
                exact_match = torch.all(predicted_tokens == shift_labels, dim=-1).item()
                
                if exact_match:
                    match_count += 1
        
        return match_count / total_prompts if total_prompts > 0 else 0.0

    def _compute_candidates_loss(
        self,
        search_batch_size: int,
        sampled_ids: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences across all prompts.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                the candidate token id sequences to evaluate
        """
        all_loss = []
        embedding_layer = self.embedding_layer

        # Debug: Check if we have valid inputs
        if sampled_ids.shape[0] == 0:
            logger.warning(
                "Empty sampled_ids tensor passed to _compute_candidates_loss"
            )
            return torch.tensor([], device=sampled_ids.device, dtype=torch.float32)

        # TODO: evaluate in practice whether the memory overhead is manageable
        # Cache optimization: store expanded caches per prompt to reuse across batches
        prompt_cache_batches = {}  # Dict[int, DynamicCache] - keyed by prompt index
        cache_batch_size = None  # Track the batch size for which caches were built

        for i in range(0, sampled_ids.shape[0], search_batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i : i + search_batch_size]
                current_batch_size = sampled_ids_batch.shape[0]

                # Compute average loss across all prompts for this batch
                total_loss = 0.0
                for prompt_idx, prompt_data in enumerate(self.prompt_data):
                    # Create input embeddings for this prompt
                    if prompt_data.get("prefix_cache"):
                        input_embeds_batch = torch.cat(
                            [
                                embedding_layer(sampled_ids_batch),
                                prompt_data["after_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                                prompt_data["target_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                            ],
                            dim=1,
                        )

                        # OPTIMIZATION: Only rebuild cache when necessary
                        # Need to rebuild if: 1) first time seeing this prompt, or 2) batch size changed
                        if (
                            cache_batch_size != current_batch_size
                            or prompt_idx not in prompt_cache_batches
                        ):
                            prefix_cache_batch = DynamicCache()
                            for layer_idx in range(len(prompt_data["prefix_cache"])):
                                key, value = prompt_data["prefix_cache"][layer_idx]
                                prefix_cache_batch.update(
                                    key.expand(current_batch_size, -1, -1, -1),
                                    value.expand(current_batch_size, -1, -1, -1),
                                    layer_idx,
                                )
                            prompt_cache_batches[prompt_idx] = prefix_cache_batch
                        else:
                            # Reuse existing expanded cache
                            prefix_cache_batch = prompt_cache_batches[prompt_idx]

                        outputs = self.model(
                            inputs_embeds=input_embeds_batch,
                            past_key_values=prefix_cache_batch,
                            use_cache=True,
                        )
                    else:
                        input_embeds_batch = torch.cat(
                            [
                                prompt_data["before_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                                embedding_layer(sampled_ids_batch),
                                prompt_data["after_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                                prompt_data["target_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                            ],
                            dim=1,
                        )
                        outputs = self.model(inputs_embeds=input_embeds_batch)

                    logits = outputs.logits

                    tmp = (
                        input_embeds_batch.shape[1] - prompt_data["target_ids"].shape[1]
                    )
                    shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                    shift_labels = prompt_data["target_ids"].repeat(
                        current_batch_size, 1
                    )

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(
                            shift_logits, -1, shift_labels.unsqueeze(-1)
                        ).squeeze(-1)
                        loss = mellowmax(
                            -label_logits, alpha=self.config.mellowmax_alpha, dim=-1
                        )
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            reduction="none",
                        )

                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                    total_loss += loss

                    # Early stopping check
                    if self._check_early_stopping(
                        prompt_data,
                        current_batch_size,
                        shift_logits=shift_logits,
                        shift_labels=shift_labels,
                        loss=loss,
                    ):
                        self.stop_flag = True

                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                # Average loss across all prompts
                avg_loss = total_loss / len(self.prompt_data)
                all_loss.append(avg_loss)

                # Update batch size tracker for cache optimization
                cache_batch_size = current_batch_size

        # Clean up cached objects to prevent memory accumulation
        prompt_cache_batches.clear()

        if not all_loss:
            # Handle edge case where no losses were computed
            return torch.tensor([], device=sampled_ids.device, dtype=torch.float32)
        return torch.cat(all_loss, dim=0)

    def _compute_candidates_loss_autoregressive(
        self,
        search_batch_size: int,
        sampled_ids: Tensor,
    ) -> Tensor:
        """Computes the GCG loss on candidate token sequences using autoregressive generation.

        This method evaluates candidates by generating tokens autoregressively (without teacher forcing),
        making it more realistic to actual inference-time behavior. It computes the negative log-likelihood
        of the target sequence under the autoregressive generation process.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            sampled_ids : Tensor, shape = (search_width, n_optim_ids)
                the candidate token id sequences to evaluate
        """
        all_loss = []
        embedding_layer = self.embedding_layer

        # Debug: Check if we have valid inputs
        if sampled_ids.shape[0] == 0:
            logger.warning(
                "Empty sampled_ids tensor passed to _compute_candidates_loss_autoregressive"
            )
            return torch.tensor([], device=sampled_ids.device, dtype=torch.float32)

        # Cache optimization: store expanded caches per prompt to reuse across batches
        prompt_cache_batches = {}  # Dict[int, DynamicCache] - keyed by prompt index
        cache_batch_size = None  # Track the batch size for which caches were built

        for i in range(0, sampled_ids.shape[0], search_batch_size):
            with torch.no_grad():
                sampled_ids_batch = sampled_ids[i : i + search_batch_size]
                current_batch_size = sampled_ids_batch.shape[0]

                # Compute average loss across all prompts for this batch
                total_loss = 0.0
                for prompt_idx, prompt_data in enumerate(self.prompt_data):
                    # Initialize loss accumulator for this prompt
                    sequence_loss = torch.zeros(
                        current_batch_size, device=sampled_ids.device
                    )

                    # Get target information
                    target_ids = prompt_data["target_ids"]
                    target_length = target_ids.shape[1]

                    # Track generated tokens for exact match checking
                    generated_tokens = torch.zeros(
                        (current_batch_size, target_length),
                        dtype=torch.long,
                        device=sampled_ids.device,
                    )

                    # Prepare initial embeddings and cache
                    if prompt_data.get("prefix_cache"):
                        # Start with optim + after embeddings (prefix is already cached)
                        current_embeds = torch.cat(
                            [
                                embedding_layer(sampled_ids_batch),
                                prompt_data["after_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                            ],
                            dim=1,
                        )

                        # Only rebuild cache when necessary
                        if (
                            cache_batch_size != current_batch_size
                            or prompt_idx not in prompt_cache_batches
                        ):
                            prefix_cache_batch = DynamicCache()
                            for layer_idx in range(len(prompt_data["prefix_cache"])):
                                key, value = prompt_data["prefix_cache"][layer_idx]
                                prefix_cache_batch.update(
                                    key.expand(current_batch_size, -1, -1, -1),
                                    value.expand(current_batch_size, -1, -1, -1),
                                    layer_idx,
                                )
                            prompt_cache_batches[prompt_idx] = prefix_cache_batch
                        else:
                            # Reuse existing expanded cache
                            prefix_cache_batch = prompt_cache_batches[prompt_idx]

                        current_cache = prefix_cache_batch
                    else:
                        # Without prefix cache, start with full prompt embeddings
                        current_embeds = torch.cat(
                            [
                                prompt_data["before_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                                embedding_layer(sampled_ids_batch),
                                prompt_data["after_embeds"].repeat(
                                    current_batch_size, 1, 1
                                ),
                            ],
                            dim=1,
                        )
                        current_cache = None

                    # Generate tokens autoregressively and compute loss at each step
                    for t in range(target_length):
                        # Forward pass with current embeddings
                        outputs = self.model(
                            inputs_embeds=current_embeds,
                            past_key_values=current_cache,
                            use_cache=True,
                        )

                        # Get logits for the last position (next token prediction)
                        logits = outputs.logits[:, -1, :]

                        # Compute log probabilities
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                        # Get the log probability of the target token at this position
                        target_token = target_ids[:, t].expand(current_batch_size)
                        token_log_probs = log_probs.gather(
                            1, target_token.unsqueeze(1)
                        ).squeeze(1)

                        # Accumulate negative log likelihood
                        sequence_loss -= token_log_probs

                        # For next step: use the predicted token (greedy decoding)
                        # This is the key difference from teacher forcing
                        next_token = logits.argmax(dim=-1)
                        next_embeds = embedding_layer(next_token).unsqueeze(1)

                        # Store generated token for exact match checking
                        generated_tokens[:, t] = next_token

                        # Update inputs for next step
                        current_embeds = next_embeds
                        current_cache = outputs.past_key_values

                    # Average loss over sequence length to get per-token loss
                    loss = sequence_loss / target_length

                    # Add this prompt's loss to the total
                    total_loss += loss

                    # Early stopping check
                    if self._check_early_stopping(
                        prompt_data,
                        current_batch_size,
                        loss=loss,
                        generated_tokens=generated_tokens,
                    ):
                        self.stop_flag = True

                    del outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                # Average loss across all prompts
                avg_loss = total_loss / len(self.prompt_data)
                all_loss.append(avg_loss)

                # Update batch size tracker for cache optimization
                cache_batch_size = current_batch_size

        # Clean up cached objects to prevent memory accumulation
        prompt_cache_batches.clear()

        if not all_loss:
            # Handle edge case where no losses were computed
            return torch.tensor([], device=sampled_ids.device, dtype=torch.float32)
        return torch.cat(all_loss, dim=0)


# A wrapper around the GCG `run` method that provides a simple API
def run_nanogcg(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages_and_targets: List[Tuple[Union[str, List[dict]], str]],
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG across multiple prompts.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages_and_targets: List of (messages, target) pairs for multi-prompt optimization.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    # Enable debug logging if debug flag is set
    if config.debug:
        logger.setLevel(logging.DEBUG)

    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages_and_targets)
    return result
