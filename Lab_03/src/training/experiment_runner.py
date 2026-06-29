"""Experiment catalog and orchestration for the laboratory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn, optim

from src.baselines.classical_todo import ClassicalBaseline
from src.config import AppConfig
from src.data.aligned_datamodule import AlignedUTKFaceDataModule
from src.data.datamodule import UTKFaceDataModule
from src.evaluation.metrics import MultiTaskEvaluator, MultiTaskMetrics
from src.evaluation.plots import ResultPlotter
from src.evaluation.reporter import ExperimentResult, ExperimentStatus
from src.models.cnn import MultiTaskCNN
from src.models.mlp_todo import MultiTaskMLP
from src.models.resnet_todo import MultiTaskResNet
from src.training.losses import MultiTaskLoss
from src.training.trainer import MultiTaskTrainer
from src.utils import set_seed


@dataclass(frozen=True)
class ExperimentSpec:
    """Configuration for one base experiment or one single-change ablation."""

    strategy_id: str
    strategy_name: str
    name: str
    variant: str
    changed_component: str
    implemented: bool
    model_kind: str
    use_augmentation: bool = True
    dropout: float = 0.4
    lambda_age: float = 0.01
    learning_rate: float = 1e-3
    # ResNet fine-tuning: number of residual blocks unfrozen from the last one.
    unfreeze_blocks: int = 0
    # E6: use raw UTKFace images processed by FaceAligner instead of pre-cropped ones.
    use_raw_data: bool = False
    # Normalize age targets to [0, ~1] (divide by 100) during training.
    # The evaluator automatically rescales back to years for reporting.
    normalize_age: bool = False
    # After training, additionally evaluate this model on the raw+FaceAligner test set.
    # Used by cnn_base to measure the domain gap quantified in E6.
    cross_eval_with_raw: bool = False
    # E1 classical baseline parameters.
    pca_components: int | float = 0.99
    gender_clf: str = "logistic"
    age_reg: str = "ridge"


def build_experiment_catalog(config: AppConfig) -> dict[str, ExperimentSpec]:
    """Return all required strategies and their expected ablation studies."""

    low_lambda = config.lambda_age / 10
    high_lambda = config.lambda_age * 10

    specs = [
        # ------------------------------------------------------------------
        # E1: classical baseline — PCA + scikit-learn estimators.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_base", "base",
            "ninguno", True, "classical",
            pca_components=0.99, gender_clf="logistic", age_reg="ridge",
        ),
        ExperimentSpec(
            "E1", "Baseline clasico", "classical_pca_low", "ablacion",
            "PCA=50 componentes (informacion insuficiente)", True, "classical",
            pca_components=50, gender_clf="logistic", age_reg="ridge",
        ),
        # ------------------------------------------------------------------
        # E2: MLP multitarea.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E2", "MLP multitarea", "mlp_base", "base",
            "ninguno", True, "mlp",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2", "MLP multitarea", "mlp_no_dropout", "ablacion",
            "dropout=0.0", True, "mlp",
            use_augmentation=True, dropout=0.0,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2", "MLP multitarea", "mlp_lambda_low", "ablacion",
            f"lambda_age={low_lambda:g}", True, "mlp",
            use_augmentation=True, dropout=0.4,
            lambda_age=low_lambda, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E2", "MLP multitarea", "mlp_lambda_high", "ablacion",
            f"lambda_age={high_lambda:g}", True, "mlp",
            use_augmentation=True, dropout=0.4,
            lambda_age=high_lambda, learning_rate=config.learning_rate,
        ),
        # ------------------------------------------------------------------
        # E3: CNN simple multitarea — fully implemented reference example.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_base", "base",
            "ninguno", True, "cnn",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            cross_eval_with_raw=True,
        ),
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_no_augmentation", "ablacion",
            "sin aumentacion", True, "cnn",
            use_augmentation=False, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_no_dropout", "ablacion",
            "dropout=0.0", True, "cnn",
            use_augmentation=True, dropout=0.0,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_lambda_low", "ablacion",
            f"lambda_age={low_lambda:g}", True, "cnn",
            use_augmentation=True, dropout=0.4,
            lambda_age=low_lambda, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_lambda_high", "ablacion",
            f"lambda_age={high_lambda:g}", True, "cnn",
            use_augmentation=True, dropout=0.4,
            lambda_age=high_lambda, learning_rate=config.learning_rate,
        ),
        ExperimentSpec(
            "E3", "CNN simple multitarea", "cnn_age_normalized", "ablacion",
            "edad normalizada a [0,1], lambda=1.0", True, "cnn",
            use_augmentation=True, dropout=0.4,
            lambda_age=1.0, learning_rate=config.learning_rate,
            normalize_age=True,
        ),
        # ------------------------------------------------------------------
        # E4: ResNet18 backbone completamente congelado.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E4", "ResNet18 congelada", "resnet_frozen_base", "base",
            "ninguno", True, "resnet_frozen",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            unfreeze_blocks=0,
        ),
        ExperimentSpec(
            "E4", "ResNet18 congelada", "resnet_frozen_no_augmentation", "ablacion",
            "sin aumentacion", True, "resnet_frozen",
            use_augmentation=False, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            unfreeze_blocks=0,
        ),
        ExperimentSpec(
            "E4", "ResNet18 congelada", "resnet_frozen_lambda_low", "ablacion",
            f"lambda_age={low_lambda:g}", True, "resnet_frozen",
            use_augmentation=True, dropout=0.4,
            lambda_age=low_lambda, learning_rate=config.learning_rate,
            unfreeze_blocks=0,
        ),
        ExperimentSpec(
            "E4", "ResNet18 congelada", "resnet_frozen_lambda_high", "ablacion",
            f"lambda_age={high_lambda:g}", True, "resnet_frozen",
            use_augmentation=True, dropout=0.4,
            lambda_age=high_lambda, learning_rate=config.learning_rate,
            unfreeze_blocks=0,
        ),
        # ------------------------------------------------------------------
        # E5: ResNet18 con fine-tuning parcial.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E5", "ResNet18 fine-tuning", "resnet_finetuning_base", "base",
            "layer4 descongelado", True, "resnet_finetuning",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            unfreeze_blocks=1,
        ),
        ExperimentSpec(
            "E5", "ResNet18 fine-tuning", "resnet_finetuning_unfreeze_more", "ablacion",
            "layer3+layer4 descongelados", True, "resnet_finetuning",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            unfreeze_blocks=2,
        ),
        ExperimentSpec(
            "E5", "ResNet18 fine-tuning", "resnet_finetuning_lr_low", "ablacion",
            "learning rate 1e-4", True, "resnet_finetuning",
            use_augmentation=True, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=1e-4,
            unfreeze_blocks=1,
        ),
        ExperimentSpec(
            "E5", "ResNet18 fine-tuning", "resnet_finetuning_lambda_high", "ablacion",
            f"lambda_age={high_lambda:g}", True, "resnet_finetuning",
            use_augmentation=True, dropout=0.4,
            lambda_age=high_lambda, learning_rate=config.learning_rate,
            unfreeze_blocks=1,
        ),
        # ------------------------------------------------------------------
        # E6: CNN entrenada sobre imágenes raw con alineación automática.
        # Contrasta con cnn_base para medir el domain gap en la app Streamlit.
        # ------------------------------------------------------------------
        ExperimentSpec(
            "E6", "CNN alineacion automatica", "cnn_aligned_base", "base",
            "datos raw alineados automaticamente, sin aumentacion", True, "cnn",
            use_augmentation=False, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            use_raw_data=True,
        ),
        ExperimentSpec(
            "E6", "CNN alineacion automatica", "resnet_aligned_base", "ablacion",
            "ResNet18 fine-tuning sobre raw+FaceAligner", True, "resnet_finetuning",
            use_augmentation=False, dropout=0.4,
            lambda_age=config.lambda_age, learning_rate=config.learning_rate,
            unfreeze_blocks=1, use_raw_data=True,
        ),
    ]
    return {spec.name: spec for spec in specs}


class ExperimentRunner:
    """Run selected experiments and preserve report rows for every strategy."""

    def __init__(
        self,
        config: AppConfig,
        device: torch.device,
        catalog: dict[str, ExperimentSpec],
    ) -> None:
        self.config = config
        self.device = device
        self.catalog = catalog
        self.plotter = ResultPlotter(config.plots_dir)

    def run(self, selected_names: set[str]) -> list[ExperimentResult]:
        unknown = selected_names.difference(self.catalog)
        if unknown:
            raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")

        results: list[ExperimentResult] = []
        for spec in self.catalog.values():
            if not spec.implemented:
                results.append(self._not_implemented_result(spec))
            elif spec.name not in selected_names:
                results.append(self._not_executed_result(spec))
            else:
                results.append(self._run_spec(spec))

        for strategy_id in ("E1", "E2", "E3", "E4", "E5", "E6"):
            self.plotter.plot_ablation_comparison(results, strategy_id)
        return results

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _run_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        print(f"\nEjecutando {spec.name}: {spec.changed_component}")
        try:
            if spec.model_kind == "classical":
                return self._run_classical_spec(spec)
            return self._run_neural_spec(spec)
        except Exception as error:
            return ExperimentResult(
                strategy_id=spec.strategy_id,
                strategy_name=spec.strategy_name,
                experiment_name=spec.name,
                variant=spec.variant,
                changed_component=spec.changed_component,
                status=ExperimentStatus.ERROR,
                message=str(error),
            )

    # ------------------------------------------------------------------
    # Neural training path (E2, E3, E4, E5, E6)
    # ------------------------------------------------------------------

    def _run_neural_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        set_seed(self.config.seed)

        data_module = self._build_data_module(spec)
        data_module.setup()

        model, model_kwargs = self._build_model(spec)
        model = model.to(self.device)

        checkpoint_path = self.config.checkpoints_dir / spec.name / "best_model.pt"
        history: list[dict] | None = None
        training_seconds = 0.0

        if checkpoint_path.exists():
            print(f"  Checkpoint encontrado, omitiendo entrenamiento.")
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(saved["model_state_dict"])
        else:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=spec.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            loss_function = MultiTaskLoss(lambda_age=spec.lambda_age)

            trainer = MultiTaskTrainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                device=self.device,
                checkpoint_path=checkpoint_path,
                checkpoint_metadata={
                    "experiment_name": spec.name,
                    "strategy_id": spec.strategy_id,
                    "model_name": spec.model_kind,
                    "model_kwargs": model_kwargs,
                    "image_size": self.config.image_size,
                    "lambda_age": spec.lambda_age,
                },
            )
            history, training_seconds = trainer.fit(
                data_module.train_dataloader(),
                data_module.val_dataloader(),
                epochs=self.config.epochs,
            )
            trainer.load_best_checkpoint()

        age_scale = 100.0 if spec.normalize_age else 1.0
        evaluator = MultiTaskEvaluator(self.device)
        evaluation = evaluator.evaluate(
            model, data_module.test_dataloader(), age_scale=age_scale
        )

        if history is not None:
            self.plotter.plot_training_history(history, spec.name)
        self.plotter.plot_confusion_matrix(evaluation, spec.name)
        self.plotter.plot_age_predictions(evaluation, spec.name)

        sizes = data_module.split_sizes()
        metrics = dict(evaluation.metrics)
        metrics.update({
            "train_samples": sizes["train"],
            "validation_samples": sizes["validation"],
            "test_samples": sizes["test"],
        })

        # Domain-gap cross-evaluation: evaluate on the raw+FaceAligner test set.
        # Used by cnn_base to produce the three numbers needed for E6 analysis.
        if spec.cross_eval_with_raw and self.config.utkface_raw_dir is not None:
            raw_dm = AlignedUTKFaceDataModule(self.config, use_augmentation=False)
            raw_dm.setup()
            raw_eval = evaluator.evaluate(model, raw_dm.test_dataloader())
            metrics.update({f"raw_{k}": v for k, v in raw_eval.metrics.items()})
            self.plotter.plot_confusion_matrix(raw_eval, f"{spec.name}_raw")
            self.plotter.plot_age_predictions(raw_eval, f"{spec.name}_raw")

        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            trainable_parameters=self._count_trainable_parameters(model),
            training_seconds=training_seconds,
            checkpoint=str(checkpoint_path),
            message="",
        )

    # ------------------------------------------------------------------
    # Classical training path (E1)
    # ------------------------------------------------------------------

    def _run_classical_spec(self, spec: ExperimentSpec) -> ExperimentResult:
        set_seed(self.config.seed)

        # Use the evaluation transform (no random augmentation) for classical
        # features; augmentation does not help sklearn estimators.
        data_module = UTKFaceDataModule(self.config, use_augmentation=False)
        data_module.setup()

        baseline = ClassicalBaseline(
            n_components=spec.pca_components,
            gender_clf=spec.gender_clf,
            age_reg=spec.age_reg,
        )

        start = time.perf_counter()
        baseline.fit_from_loader(data_module.train_dataloader())
        training_seconds = time.perf_counter() - start

        gender_preds, age_preds = baseline.predict_from_loader(data_module.test_dataloader())
        y_gender, y_age = baseline.targets_from_loader(data_module.test_dataloader())

        import torch as _torch
        evaluation = MultiTaskMetrics.calculate(
            gender_targets=_torch.tensor(y_gender, dtype=_torch.long),
            gender_predictions=_torch.tensor(gender_preds, dtype=_torch.long),
            age_targets=_torch.tensor(y_age, dtype=_torch.float32),
            age_predictions=_torch.tensor(age_preds, dtype=_torch.float32),
        )

        # Plot confusion matrix and age scatter (no training curves for classical).
        self.plotter.plot_confusion_matrix(evaluation, spec.name)
        self.plotter.plot_age_predictions(evaluation, spec.name)

        sizes = data_module.split_sizes()
        metrics = dict(evaluation.metrics)
        metrics.update({
            "train_samples": sizes["train"],
            "validation_samples": sizes["validation"],
            "test_samples": sizes["test"],
        })
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            trainable_parameters=None,
            training_seconds=training_seconds,
            checkpoint="",
            message="",
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    def _build_data_module(self, spec: ExperimentSpec) -> UTKFaceDataModule:
        if spec.use_raw_data:
            return AlignedUTKFaceDataModule(
                self.config,
                use_augmentation=spec.use_augmentation,
                normalize_age=spec.normalize_age,
            )
        return UTKFaceDataModule(
            self.config,
            use_augmentation=spec.use_augmentation,
            normalize_age=spec.normalize_age,
        )

    def _build_model(self, spec: ExperimentSpec) -> tuple[nn.Module, dict]:
        if spec.model_kind == "cnn":
            kwargs: dict = {"dropout": spec.dropout}
            return MultiTaskCNN(**kwargs), kwargs

        if spec.model_kind == "mlp":
            kwargs = {
                "image_size": self.config.image_size,
                "dropout": spec.dropout,
            }
            return MultiTaskMLP(**kwargs), kwargs

        if spec.model_kind in ("resnet_frozen", "resnet_finetuning"):
            kwargs = {
                "unfreeze_blocks": spec.unfreeze_blocks,
                "dropout": spec.dropout,
            }
            return MultiTaskResNet(**kwargs), kwargs

        raise NotImplementedError(
            f"No existe una fabrica para model_kind={spec.model_kind!r}."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_trainable_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _not_implemented_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_IMPLEMENTED,
            message="El experimento debe ser completado por los alumnos.",
        )

    @staticmethod
    def _not_executed_result(spec: ExperimentSpec) -> ExperimentResult:
        return ExperimentResult(
            strategy_id=spec.strategy_id,
            strategy_name=spec.strategy_name,
            experiment_name=spec.name,
            variant=spec.variant,
            changed_component=spec.changed_component,
            status=ExperimentStatus.NOT_EXECUTED,
            message="Implementado, pero no fue seleccionado en esta ejecucion.",
        )
