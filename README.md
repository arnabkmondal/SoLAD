# SoLAD
SoLAD: Sampling over Latent Adapter for Few Shot Generative Modelling

1. Download FFHQ checkpoint using the following link (This is a third party link (owner: Kim Seonghyeon) and does not reveal author identity): https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO/view

2. Place the checkpoint in the folder checkpoint.

3. Download the entire babies dataset for FID computation using the following link (This is a third party link (owner: Utkarsh Ojha) and does not reveal author identity): https://drive.google.com/file/d/1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID/view?usp=sharing

4. Create a folder ./data/real_babies and extract all the real images in that directory.

5. Create an environment as per the specifications of requirements.yml

6. Activate the newly created environment.

7. Change directory to 10shot_babies

8. Run babies_job.sh (Note that you might need to comment out the first two lines of the babies_job.sh i.e., module load compiler/cuda/10.0/compilervars and
module load compiler/gcc/6.5.0/compilervars. However, the code requires CUDA 10.0 and GCC 6.5.0 for successful execution)
