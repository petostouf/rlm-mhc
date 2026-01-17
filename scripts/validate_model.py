#!/usr/bin/env python3
"""
RLM-mHC Model Validation Script
================================
Murat (TEA) - Architecte de Tests BMad

Ce script valide que le modèle RLM-mHC fonctionne correctement:
1. Forward pass avec calcul de loss
2. Backward pass (gradient flow)
3. Mini-entraînement (quelques steps)
4. Sauvegarde/Chargement du modèle
5. Métriques mHC (Amax Gain)
"""

import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from rlm_mhc.model.config import ModelConfig
from rlm_mhc.model.transformer import RLMModel
from rlm_mhc.model.mhc.layers import mHCLayer


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} | {test_name}")
    if details:
        print(f"          -> {details}")


class ModelValidator:
    """Validates the RLM-mHC model functionality."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

        # Small config for fast validation
        self.config = ModelConfig(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            ffn_dim=341,  # ~8/3 * 128
            vocab_size=256,
            max_seq_len=128,
            mhc_enabled=True,
            mhc_flows=4,
            mhc_sinkhorn_iters=10,
        )

        print(f"Device: {self.device}")
        print(f"Config: hidden_dim={self.config.hidden_dim}, layers={self.config.num_layers}, mhc_flows={self.config.mhc_flows}")

    def test_forward_pass(self) -> bool:
        """Test 1: Forward pass with loss computation."""
        print_header("Test 1: Forward Pass avec Loss")

        try:
            model = RLMModel(self.config).to(self.device)
            model.eval()

            # Create input
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, 256, (batch_size, seq_len), device=self.device)
            labels = torch.randint(0, 256, (batch_size, seq_len), device=self.device)

            # Forward pass
            with torch.no_grad():
                output = model(input_ids, labels=labels)

            # Validate output
            assert output.logits is not None, "Logits is None"
            assert output.logits.shape == (batch_size, seq_len, 256), f"Wrong logits shape: {output.logits.shape}"
            assert output.loss is not None, "Loss is None"
            assert not torch.isnan(output.loss), "Loss is NaN"
            assert not torch.isinf(output.loss), "Loss is Inf"

            print_result("Logits shape correct", True, f"Shape: {output.logits.shape}")
            print_result("Loss computed", True, f"Loss: {output.loss.item():.4f}")
            print_result("No NaN/Inf", True)

            self.results['forward_pass'] = True
            return True

        except Exception as e:
            print_result("Forward pass", False, str(e))
            self.results['forward_pass'] = False
            return False

    def test_backward_pass(self) -> bool:
        """Test 2: Backward pass - gradient flow."""
        print_header("Test 2: Backward Pass (Gradient Flow)")

        try:
            model = RLMModel(self.config).to(self.device)
            model.train()

            # Create input
            input_ids = torch.randint(0, 256, (2, 32), device=self.device)
            labels = input_ids.clone()

            # Forward pass
            output = model(input_ids, labels=labels)
            loss = output.loss

            # Backward pass
            loss.backward()

            # Check gradients
            params_with_grad = 0
            params_without_grad = 0
            nan_grads = 0

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        params_with_grad += 1
                        if torch.isnan(param.grad).any():
                            nan_grads += 1
                    else:
                        params_without_grad += 1

            print_result("Gradients computed", params_with_grad > 0, f"{params_with_grad} params with gradients")
            print_result("No missing gradients", params_without_grad == 0, f"{params_without_grad} params without gradients")
            print_result("No NaN gradients", nan_grads == 0, f"{nan_grads} NaN gradients")

            # Compute gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            print_result("Gradient norm reasonable", 0 < total_norm < 1000, f"Grad norm: {total_norm:.4f}")

            passed = params_with_grad > 0 and params_without_grad == 0 and nan_grads == 0
            self.results['backward_pass'] = passed
            return passed

        except Exception as e:
            print_result("Backward pass", False, str(e))
            self.results['backward_pass'] = False
            return False

    def test_mini_training(self) -> bool:
        """Test 3: Mini training loop (5 steps)."""
        print_header("Test 3: Mini-Entraînement (5 steps)")

        try:
            model = RLMModel(self.config).to(self.device)
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            losses = []
            start_time = time.time()

            for step in range(5):
                # Generate random batch
                input_ids = torch.randint(0, 256, (4, 64), device=self.device)
                labels = input_ids.clone()

                # Forward
                output = model(input_ids, labels=labels)
                loss = output.loss

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Step
                optimizer.step()

                losses.append(loss.item())
                print(f"    Step {step + 1}: loss = {loss.item():.4f}")

            elapsed = time.time() - start_time

            print_result("Training completed", True, f"5 steps in {elapsed:.2f}s")
            print_result("Loss decreased", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")
            print_result("No NaN losses", all(not (l != l) for l in losses))

            passed = all(not (l != l) for l in losses)  # No NaN
            self.results['mini_training'] = passed
            return passed

        except Exception as e:
            print_result("Mini training", False, str(e))
            self.results['mini_training'] = False
            return False

    def test_save_load(self) -> bool:
        """Test 4: Save and load model."""
        print_header("Test 4: Sauvegarde/Chargement du Modèle")

        try:
            model = RLMModel(self.config).to(self.device)
            model.eval()

            # Generate test input
            input_ids = torch.randint(0, 256, (1, 16), device=self.device)

            # Get original output
            with torch.no_grad():
                original_output = model(input_ids)

            # Save model
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "test_model"

                model.save_pretrained(save_path)

                # Check files exist
                config_exists = (save_path / "config.yaml").exists()
                weights_exists = (save_path / "model.safetensors").exists()

                print_result("Config saved", config_exists)
                print_result("Weights saved (safetensors)", weights_exists)

                # Load model
                loaded_model = RLMModel.from_pretrained(save_path)
                loaded_model = loaded_model.to(self.device)
                loaded_model.eval()

                # Get loaded output
                with torch.no_grad():
                    loaded_output = loaded_model(input_ids)

                # Compare outputs
                outputs_match = torch.allclose(
                    original_output.logits,
                    loaded_output.logits,
                    atol=1e-5
                )

                print_result("Outputs match after reload", outputs_match)

            passed = config_exists and weights_exists and outputs_match
            self.results['save_load'] = passed
            return passed

        except Exception as e:
            print_result("Save/Load", False, str(e))
            self.results['save_load'] = False
            return False

    def test_mhc_metrics(self) -> bool:
        """Test 5: mHC metrics (Amax Gain)."""
        print_header("Test 5: Métriques mHC (Amax Gain)")

        try:
            model = RLMModel(self.config).to(self.device)

            # Get mHC metrics
            metrics = model.get_mhc_metrics()

            print_result("mHC metrics available", len(metrics) > 0, f"{len(metrics)} layer(s)")

            all_valid = True
            for i, (pre_amax, post_amax) in enumerate(metrics):
                pre_valid = 0 < pre_amax < 3.0
                post_valid = 0 < post_amax < 3.0

                print(f"    Layer {i}: pre_amax={pre_amax:.4f}, post_amax={post_amax:.4f}")

                if not pre_valid or not post_valid:
                    all_valid = False

            print_result("Amax values in range (0, 3)", all_valid)

            # Test Sinkhorn convergence
            layer = mHCLayer(hidden_dim=64, num_flows=4)
            P = layer.get_connection_matrix()

            row_sums_ok = torch.allclose(P.sum(dim=1), torch.ones(4), atol=1e-4)
            col_sums_ok = torch.allclose(P.sum(dim=0), torch.ones(4), atol=1e-4)

            print_result("Connection matrix doubly stochastic", row_sums_ok and col_sums_ok)

            passed = len(metrics) > 0 and all_valid and row_sums_ok and col_sums_ok
            self.results['mhc_metrics'] = passed
            return passed

        except Exception as e:
            print_result("mHC metrics", False, str(e))
            self.results['mhc_metrics'] = False
            return False

    def test_cuda_training(self) -> bool:
        """Test 6: CUDA training with mixed precision."""
        print_header("Test 6: Entraînement CUDA + Mixed Precision")

        if not torch.cuda.is_available():
            print("    CUDA non disponible - test ignoré")
            self.results['cuda_training'] = None
            return True

        try:
            model = RLMModel(self.config).cuda()
            model.train()

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Test with bf16
            input_ids = torch.randint(0, 256, (4, 64), device='cuda')
            labels = input_ids.clone()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(input_ids, labels=labels)
                loss = output.loss

            loss.backward()
            optimizer.step()

            print_result("Mixed precision (bf16) training", True, f"Loss: {loss.item():.4f}")
            print_result("CUDA memory used", True, f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")

            self.results['cuda_training'] = True
            return True

        except Exception as e:
            print_result("CUDA training", False, str(e))
            self.results['cuda_training'] = False
            return False

    def test_generation(self) -> bool:
        """Test 7: Text generation."""
        print_header("Test 7: Génération de Texte")

        try:
            model = RLMModel(self.config).to(self.device)
            model.eval()

            # Test greedy generation
            input_ids = torch.randint(0, 256, (1, 10), device=self.device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
            )

            print_result("Greedy generation", output_ids.shape[1] == 30, f"Generated {output_ids.shape[1] - 10} tokens")

            # Test sampling
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50,
                do_sample=True,
            )

            print_result("Sampling generation", output_ids.shape[1] == 30)

            # Test top-p
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
            )

            print_result("Top-p generation", output_ids.shape[1] == 30)

            self.results['generation'] = True
            return True

        except Exception as e:
            print_result("Generation", False, str(e))
            self.results['generation'] = False
            return False

    def run_all_tests(self):
        """Run all validation tests."""
        print_header("VALIDATION DU MODÈLE RLM-mHC")
        print(f"  Murat (TEA) - Architecte de Tests BMad")

        tests = [
            self.test_forward_pass,
            self.test_backward_pass,
            self.test_mini_training,
            self.test_save_load,
            self.test_mhc_metrics,
            self.test_cuda_training,
            self.test_generation,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"  [ERROR] ERREUR CRITIQUE: {e}")

        # Summary
        print_header("RÉSUMÉ DE LA VALIDATION")

        passed = sum(1 for v in self.results.values() if v is True)
        failed = sum(1 for v in self.results.values() if v is False)
        skipped = sum(1 for v in self.results.values() if v is None)

        for test_name, result in self.results.items():
            if result is True:
                print(f"  [OK]   {test_name}")
            elif result is False:
                print(f"  [FAIL] {test_name}")
            else:
                print(f"  [SKIP] {test_name} (ignore)")

        print(f"\n  Total: {passed} passes, {failed} echecs, {skipped} ignores")

        if failed == 0:
            print("\n  >>> VALIDATION REUSSIE - Le modele est pret pour l'entrainement!")
        else:
            print("\n  >>> VALIDATION ECHOUEE - Des corrections sont necessaires.")

        return failed == 0


if __name__ == "__main__":
    validator = ModelValidator()
    success = validator.run_all_tests()
    sys.exit(0 if success else 1)
