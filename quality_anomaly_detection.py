import time
import gc
import torch
import pandas as pd
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore, EfficientAd
from anomalib.engine import Engine
ROOT = './datasets/MVTecAD'
CATEGORIES = ['metal_nut', 'screw', 'transistor', 'capsule']

# FPS 측정 함수
def measure_fps(model, dm, n=100):
    dm.setup()
    images = next(iter(dm.test_dataloader())).image.cuda()
    model = model.cuda().eval()
    
    with torch.no_grad():
        for _ in range(10): model(images)  
    torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): model(images)
    torch.cuda.synchronize()
    return round(n * images.shape[0] / (time.perf_counter() - t0), 1)

def main():
    results = []
    print("카테고리 자동화 벤치마크 시작 \n")

    for category in CATEGORIES:
        print(f"\n현재 카테고리: {category.upper()}")
        
        # PatchCore 학습 및 평가
        print(f"[{category}] PatchCore 진행 중...")
        dm_pc = MVTecAD(root=ROOT, category=category, train_batch_size=32, eval_batch_size=32)
        model_pc = Patchcore(backbone='wide_resnet50_2', layers=['layer2', 'layer3'], coreset_sampling_ratio=0.1, num_neighbors=9)
        engine_pc = Engine(accelerator='gpu', devices=1, default_root_dir=f'./outputs/patchcore/{category}')
        
        torch.cuda.reset_peak_memory_stats()
        engine_pc.fit(model=model_pc, datamodule=dm_pc)
        pc_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        pc_results = engine_pc.test(model=model_pc, datamodule=dm_pc)
        pc_fps = measure_fps(model_pc, dm_pc)
        
        results.append({
            'category': category, 'model': 'PatchCore', 
            'image_auroc': pc_results[0]['image_AUROC'], 
            'pixel_auroc': pc_results[0]['pixel_AUROC'], 
            'fps': pc_fps, 'gpu_mb': pc_gpu_mem
        })

        # EfficientAD-S 학습 및 평가
        print(f"[{category}] EfficientAD-S 진행 중...")
        dm_ead = MVTecAD(root=ROOT, category=category, train_batch_size=1, eval_batch_size=32)
        model_ead = EfficientAd(model_size='small')
        engine_ead = Engine(accelerator='gpu', devices=1, max_epochs=70, default_root_dir=f'./outputs/efficientad/{category}')
        
        torch.cuda.reset_peak_memory_stats()
        engine_ead.fit(model=model_ead, datamodule=dm_ead)
        ead_gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        ead_results = engine_ead.test(model=model_ead, datamodule=dm_ead)
        ead_fps = measure_fps(model_ead, dm_ead)
        
        results.append({
            'category': category, 'model': 'EfficientAD-S', 
            'image_auroc': ead_results[0]['image_AUROC'], 
            'pixel_auroc': ead_results[0]['pixel_AUROC'], 
            'fps': ead_fps, 'gpu_mb': ead_gpu_mem
        })
        
        del dm_pc, model_pc, engine_pc, dm_ead, model_ead, engine_ead
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{category} 완료 및 GPU 메모리 확보 완료")

    # 전체 결과 처리 및 저장
    df = pd.DataFrame(results)

    print('\n=== 전체 결과 (Raw) ===')
    print(df.to_string(index=False))

    if len(df) > 0:
        summary = (
            df.groupby('model')[['image_auroc', 'pixel_auroc', 'fps', 'gpu_mb']]
            .mean().round(4)
            .reindex(['PatchCore', 'EfficientAD-S'])
        )
        summary.columns = ['Image AUROC', 'Pixel AUROC', 'FPS', 'GPU Mem (MB)']
        
        print('\n=== 논문 Table  ===')
        print(summary.to_string())

        Path('results').mkdir(exist_ok=True)
        df.to_csv('results/raw.csv', index=False)
        summary.to_csv('results/summary.csv')
        print('\n저장 완료: results/raw.csv / results/summary.csv')

if __name__ == "__main__":
    main()