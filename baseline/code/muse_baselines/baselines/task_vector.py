"""Task-vector construction and weight arithmetic for unlearning."""

import torch
from transformers import AutoModelForCausalLM


def load_model(model_dir: str, **kwargs) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, **kwargs
    )


def unlearn(
    model_dir: str,
    out_dir: str | None = None,
    some_pt_model_dir: str | None = None,
    some_ft_model_dir: str | None = None,
    alpha: float = 1.0,
):
    """Subtract a scaled fine-tuning task vector from the target model."""
    if some_pt_model_dir is None or some_ft_model_dir is None:
        raise ValueError(
            "Task vector (ilharco2023) requires some pretrained & finetuned models!"
        )

    task_vector = TaskVector(
        pretrained_state_dict=load_model(some_pt_model_dir).state_dict(),
        finetuned_state_dict=load_model(some_ft_model_dir).state_dict(),
    )

    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    neg_task_vector = -task_vector

    model = load_model(model_dir)
    new_state_dict = neg_task_vector.apply_to(
        pretrained_model=model, scaling_coef=alpha, in_place=False
    )
    del model
    new_model = load_model(model_dir, state_dict=new_state_dict, device_map="auto")

    if out_dir is not None:
        new_model.save_pretrained(out_dir)
    return new_model


class TaskVector:
    """Parameter differences between a base model and a fine-tuned model."""

    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        pretrained_state_dict=None,
        finetuned_state_dict=None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passing in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            ) or (
                pretrained_state_dict is not None and finetuned_state_dict is not None
            )
            with torch.no_grad():
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(
                        pretrained_checkpoint
                    ).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self):
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        if in_place:
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict
