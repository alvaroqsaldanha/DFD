# Exploring Self-Supervised Vision Transformers for Deepfake Detection

Implementation of the paper:  <a href="https://arxiv.org/abs/2405.00355">Exploring Self-Supervised Vision Transformers for Deepfake Detection: A Comparative Analysis</a> (IJCB 2024).

You can clone this repository into your favorite directory:

    $ git clone https://github.com/nii-yamagishilab/ssl_vits_df.git

## 1. Main requirements
- pytorch 2.0.1
- torchvision 0.15.2
- scikit-learn 1.3.0
- scipy 1.12.0
- numpy 1.23.5
- timm 0.9.12
- tqdm

## 2. Project organization
1. Approach 1 - Using adaptors:

        adaptors
1. Approach 2 - Fine-tuning:

        finetuning
1. About `DataListDataset` (`datalist.py`)
   
   Please put each annotation per row as:
   
        <relative_path>/image_file,int_label
   
## 3. Authors
- Huy H. Nguyen (https://researchmap.jp/nhhuy/?lang=english)
- Junichi Yamagishi (https://researchmap.jp/read0205283/?lang=english)
- Isao Echizen (https://researchmap.jp/echizenisao/?lang=english)

## Acknowledgement
This work was partially supported by JSPS KAKENHI Grants JP21H04907 and JP24H00732, by JST CREST Grants JPMJCR18A6 and JPMJCR20D3 including AIP challenge program, by JST AIP Acceleration Grant JPMJCR24U3 Japan, and by the project for the development and demonstration of countermeasures against disinformation and misinformation on the Internet with the Ministry of Internal Affairs and Communications of Japan.

## Reference
H. H. Nguyen, J. Yamagishi, and I. Echizen, “Exploring Self-Supervised Vision Transformers for Deepfake Detection: A Comparative Analysis,” IEEE International Joint Conference on Biometrics (IJCB) 2024.
