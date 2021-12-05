# scRNA-seq-celltype-and-tissue-markers

# Data

Download the labeled data we use in the study from Google Drive
- Human data (processed via HuBMAP pipeline):
  - We ran a data integration and label transfer pipeline to label this data, which you can run yourself, see the code and README under "labeling_pipeline"
  - Or you can just donwload the end result labeled data files [here](https://drive.google.com/drive/folders/1g8jkxTm0FWPpyhAztVeVbW0v4atepaJ-?usp=sharing), and place them under "/data/hubmap/" (so that you will have "data/hubmap/thymus", "data/hubmap/spleen" etc.)
- Human PBMC data (processed by the original authors):
  - You can get this from the original paper, but we have re-uploaded those same files [here](https://drive.google.com/drive/folders/1vR3A_zakd2yOXFyEAzi6z3cdKcdzaXTA?usp=sharing) on Google Drive to guarantee access. Place them under "data/van_der_Wijst_PBMC/"
- Tabula Muris mouse data:
  - Again, you can get this form the original paper, but we have re-uploaded those same files [here](https://drive.google.com/drive/folders/15dXhaVam976sGofBxYo94SmC8ogFHqi7?usp=sharing) on Google Drive to guarantee access. Place them under "data/tabula_muris"


# Results (cell type and tissue type marker lists)

You can see the final marker lists for Human cell type and tissue combinations from our Exclusive L1 approach in "markers.csv". See next section for running the analysis yourself.

# Running the analysis pipeline
Our methods code are all in "find_markers.ipynb". Once the data from "Data" above are all donwloaded and placed in their proper subfolders in the "data" directory, you can simply run this entire notebook and it will train the models and generate the figures like we had in our paper.

