CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/flower/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/flower --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/trex/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/trex --images images_4
