#figurines
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/lerf_data/figurines --images images --model_path output/lerf/figurines
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/lerf/figurines/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images

CUDA_VISIBLE_DEVICES=1 python viewer.py --gs_source output/lerf/figurines/point_cloud/iteration_10000/point_cloud.ply  --colmap_dir data/lerf_data/figurines --feature_gs_source output/lerf/figurines/point_cloud/iteration_10000/16_feature_gs_10000.pt --images images
CUDA_VISIBLE_DEVICES=1 python render.py --cfg_path scripts/lerf_render.json  --scene figurines --images images --clip --lisa
#teatime
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/lerf_data/teatime --images images --model_path output/lerf/teatime
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/lerf/teatime/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lerf_data/teatime --images images