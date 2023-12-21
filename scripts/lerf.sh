#figurines
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/lerf_data/figurines --images images --iterations 5000 --model_path output/lerf/figurines
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/lerf/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images

CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/lerf/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/lerf/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/lerf/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images --global_contrastive
python viewer.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/fern --feature_gs_source output/llff/fern/point_cloud/iteration_5000/feature_gs_20000.pt --images images_4
python test.py --cfg_path scripts/batch_contrastive_llff_test_config.json  --scene fern --images images_4
python test.py --cfg_path scripts/queue_contrastive_llff_test_config.json  --scene fern --images images_4
python test.py --cfg_path scripts/global_contrastive_llff_test_config.json  --scene fern --images images_4
