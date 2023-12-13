#garden
python train_gs.py --gs_source output/garden/point_cloud/iteration_5000/point_cloud.ply --images images_4 --iterations 5000
python train.py --gs_source output/garden/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/360_v2/garden --images images_4
python viewer.py --gs_source output/garden/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/360_v2/garden --feature_gs_source output/garden/point_cloud/iteration_5000/feature_gs_20000.pt
python viewer.py --gs_source output/kitchen/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/360_v2/kitchen --feature_gs_source output/kitchen/point_cloud/iteration_5000/feature_gs_10000.pt