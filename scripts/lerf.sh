#figurines
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --images images
python viewer.py --gs_source output/figurines/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/lerf_data/figurines --feature_gs_source output/figurines/point_cloud/iteration_5000/feature_gs_20000.pt
