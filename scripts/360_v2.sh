#garden
python train_gs.py --gs_source output/360_v2/garden/point_cloud/iteration_5000/point_cloud.ply --images images_4 --iterations 5000
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/360_v2/garden/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/360_v2/garden --images images_4
CUDA_VISIBLE_DEVICES=2 python viewer.py --gs_source output/360_v2/garden/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/360_v2/garden --feature_gs_source output/360_v2/garden/point_cloud/iteration_5000/feature_gs_10000.pt --gemini
python test.py --cfg_path scripts/360_v2_test_config.json  --scene garden --images images_4