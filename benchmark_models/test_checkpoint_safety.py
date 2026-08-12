import os
import sys
import torch
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))
from train_auraseg_r18_wacv import Config, AURASegTrainer_R18

@dataclass
class DummyArgs:
    fusion_type: str = 'mul'
    attention_mode: str = 'full'
    use_sobel: bool = True
    use_gate: bool = True
    seed: int = 42
    output_root: str = ''
    resume_from: str = None

def run_tests():
    print("Running Checkpoint Safety & Resume Tests...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        args = DummyArgs(output_root=str(tmp_path))
        
        # Override disk usage check temporarily for the test environment
        original_disk_usage = shutil.disk_usage
        shutil.disk_usage = lambda path: original_disk_usage("/") # Fake it for test
        
        try:
            config = Config(args)
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (config.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
            
            device = torch.device('cpu')
            trainer = AURASegTrainer_R18(config, device)
            
            print("1. Testing Atomic Save...")
            dummy_metrics = {'miou': 0.85}
            trainer.best_miou = 0.80
            
            # Save latest and best
            trainer.save_checkpoint(epoch=5, metrics=dummy_metrics, is_best=True)
            
            latest_path = config.OUTPUT_DIR / "checkpoints" / "latest.pth"
            best_path = config.OUTPUT_DIR / "checkpoints" / "best.pth"
            
            assert latest_path.exists(), "latest.pth not found!"
            assert best_path.exists(), "best.pth not found!"
            assert not latest_path.with_suffix('.pth.tmp').exists(), "tmp file not cleaned up!"
            
            print("[SUCCESS] Atomic save generated files correctly.")
            
            print("2. Testing Loading Checkpoint...")
            loaded_ckpt = torch.load(latest_path, weights_only=False)
            assert loaded_ckpt['epoch'] == 5, "Epoch mismatch"
            assert loaded_ckpt['metrics']['miou'] == 0.85, "Metrics mismatch"
            print("[SUCCESS] Checkpoint loaded correctly.")
            
            print("3. Testing Resume-State Restoration Structurally...")
            args_resume = DummyArgs(output_root=str(tmp_path), resume_from=str(latest_path))
            config_resume = Config(args_resume)
            
            resume_trainer = AURASegTrainer_R18(config_resume, device)
            
            assert resume_trainer.start_epoch == 6, f"Expected start_epoch=6, got {resume_trainer.start_epoch}"
            assert resume_trainer.best_miou == 0.80, f"Expected best_miou=0.80, got {resume_trainer.best_miou}"
            print("[SUCCESS] Resume state restored correctly.")
            
        finally:
            shutil.disk_usage = original_disk_usage
            
    print("\nAll structural safety tests passed!")

if __name__ == "__main__":
    run_tests()
