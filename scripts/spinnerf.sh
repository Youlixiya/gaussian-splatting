#fern
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/fern --images images_4 --iterations 5000 --model_path output/llff/fern
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --queue_contrastive
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fern --images images_4 --global_contrastive
python viewer.py --gs_source output/llff/fern/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/fern --feature_gs_source output/llff/fern/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fern --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fern --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fern --images images_4

#fortress
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/fortress --images images_4 --iterations 5000 --model_path output/llff/fortress
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/fortress --images images_4
python viewer.py --gs_source output/llff/fortress/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/fortress --feature_gs_source output/llff/fortress/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fortress --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fortress --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene fortress --images images_4

#horns
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/horns --images images_4 --iterations 5000 --model_path output/llff/horns
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/horns --images images_4
python viewer.py --gs_source output/llff/horns/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/horns --feature_gs_source output/llff/horns/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene horns --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene horns --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene horns --images images_4

#leaves
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/leaves --images images_4 --iterations 5000 --model_path output/llff/leaves
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/leaves --images images_4
python viewer.py --gs_source output/llff/leaves/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/leaves --feature_gs_source output/llff/leaves/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene leaves --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene leaves --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene leaves --images images_4

#orchids
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/orchids --images images_4 --iterations 5000 --model_path output/llff/orchids 
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply --colmap_dir data/nerf_llff_data/orchids --images images_4 
python viewer.py --gs_source output/llff/orchids/point_cloud/iteration_5000/point_cloud.ply  --colmap_dir data/nerf_llff_data/orchids --feature_gs_source output/llff/orchids/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene orchids --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene orchids --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene orchids --images images_4

#room
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/nerf_llff_data/room --images images_4 --iterations 5000 --model_path output/llff/room
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_llff_data/room --images images_4 --no_cache
python viewer.py --gs_source output/llff/room/point_cloud/iteration_10000/point_cloud.ply  --colmap_dir data/nerf_llff_data/room --feature_gs_source output/llff/room/point_cloud/iteration_5000/batch_contrastive_feature_gs_10000.pt --gemini --images images_4
python test_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json  --scene room --images images_4

#pinecone
CUDA_VISIBLE_DEVICES=2 python train_gs.py --source_path data/nerf_real_360/pinecone --images images_8 --model_path output/nerf_real_360/pinecone
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/nerf_real_360/pinecone/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_real_360/pinecone --images images_8
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/nerf_real_360/pinecone/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_real_360/pinecone --images images_8 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/nerf_real_360/pinecone/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_real_360/pinecone --images images_8 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/nerf_real_360/pinecone/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/nerf_real_360/pinecone --images images_8 --no_cache

#truck
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/tandt/truck --images images --model_path output/tandt/truck
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/tandt/truck/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/tandt/truck --images images
CUDA_VISIBLE_DEVICES=1 python train.py --gs_source output/tandt/truck/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/tandt/truck --images images --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/tandt/truck/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/tandt/truck --images images --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/tandt/truck/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/tandt/truck --images images --no_cache

#fork
CUDA_VISIBLE_DEVICES=1 python train_gs.py --source_path data/fork --images images_8 --model_path output/fork
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/fork/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/fork --images images_8
CUDA_VISIBLE_DEVICES=2 python train.py --gs_source output/fork/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/fork --images images_8 --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/fork/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/fork --images images_8 --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=4 python train.py --gs_source output/fork/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/fork --images images_8 --no_cache

#lego_real_night_radial
CUDA_VISIBLE_DEVICES=2 python train_gs.py --source_path data/lego_real_night_radial --images images --model_path output/lego_real_night_radial
CUDA_VISIBLE_DEVICES=0 python train.py --gs_source output/lego_real_night_radial/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lego_real_night_radial --images images
CUDA_VISIBLE_DEVICES=3 python train.py --gs_source output/lego_real_night_radial/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lego_real_night_radial --images images --gs_feature_dim 8
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/lego_real_night_radial/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lego_real_night_radial --images images --gs_feature_dim 32
CUDA_VISIBLE_DEVICES=5 python train.py --gs_source output/lego_real_night_radial/point_cloud/iteration_10000/point_cloud.ply --colmap_dir data/lego_real_night_radial --images images --no_cache

python metrics_spinnerf.py --cfg_path scripts/batch_contrastive_spinnerf_test_config.json --gt_path data/mvseg --save_tag spinnerf_batch_contrastive