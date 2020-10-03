# Grounding Language Attributes to Objects using Bayesian Eigenobjects
Code and the dataset for reproducing the experiments of the paper “Grounding Language Attributes to Objects using Bayesian Eigenobjects.”

## Setup
The repository needs several large dataset files, which can be downloaded here:
https://drive.google.com/drive/folders/1_6AdIbaEpOdvTo2kg4GG9z8ApKRVCs1i

## Training/
nlmodel.py
Contains the language grounding models (Bag-of-Words & Embedding Model). EmbedModel was the language model used in the paper.

nlretnn.py
Contains the training and evaluation script for the full-view experiment (with fully-observed BEO vectors).

```
usage: nlretnn.py [-h] [--beo_size BEO_SIZE] [--traindata TRAINDATA]
                  [--testdata TESTDATA] [--devdata DEVDATA]
                  [--objvectors OBJVECTORS] [--model_output MODEL_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --beo_size BEO_SIZE   size of each image dimension
  --traindata TRAINDATA
                        data file
  --testdata TESTDATA   data file
  --devdata DEVDATA     data file
  --objvectors OBJVECTORS
                        beo vectors
  --model_output MODEL_OUTPUT
                        model output name
```

nlretnn_partial.py
Contains the training and evaluation script for the partial-view and view-transfer experiments (with partially-observed BEO vectors).

```
usage: nlretnn_partial.py [-h] [--beo_size BEO_SIZE] [--traindata TRAINDATA]
                          [--testdata TESTDATA] [--devdata DEVDATA]
                          [--testvectors TESTVECTORS]
                          [--trainvectors TRAINVECTORS]
                          [--model_output MODEL_OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  --beo_size BEO_SIZE   size of each image dimension
  --traindata TRAINDATA
                        data file
  --testdata TESTDATA   data file
  --devdata DEVDATA     data file
  --testvectors TESTVECTORS
                        beo vectors
  --trainvectors TRAINVECTORS
                        beo vectors
  --model_output MODEL_OUTPUT
                        model output name
```

## Data/
Datasets are organized into car, couch, and plane respectively. Data files are equivalent across the object classes.

### car/
Contains BEO and language annotation data for each object in the cars class.

#### BEO vector files:
Format:
object_id, numpy_float_array

**car_full_obv_vecs_300.csv**
Contains the fully-observed BEO vectors for each object in the dataset.

**limited_viewpoint_car_partial_view_train.csv**
Contains BEO vectors obtained from partially observed front-facing views of objects using the techniques in “Hybrid Bayesian Eigenobjects: Combining Linear Subspace and Deep Network Methods for 3D Robot Vision” (HBEO)

**limited_viewpoint_car_partial_view_test.csv**
Contains a disjoint test-set of BEO vectors obtained for objects from side-rear-facing views of objects using HBEO.

**partial_view_car_vectors_300.csv**
Contains BEO vectors obtained from partial observations of objects, from all angles using HBEO.

**partial_view_test_car_vectors_300.csv**
Contains a disjoint test-set of BEO vectors for objects obtained from partial observations of objects, from all angles using HBEO.

#### Language annotation files:
Format:
object_id, attr_1, attr_ 2, attr_3, attr_4, attr_5, attr_6, natural_language_description

**car_train.csv**

**car_dev.csv**

**car_test.csv**

Contains 10 annotations per object_id, with attribute ratings from 1-5, and a natural language description of the object. Objects are sourced from the shapenet.org project.

## Citing
If you use our dataset or code please cite:
```
@inproceedings{cohen2019grounding,
  title={Grounding Language Attributes to Objects using Bayesian Eigenobjects},
  author={Cohen*, Vanya and Burchfiel*, Benjamin and Nguyen*, Thao and Gopalan, Nakul and Tellex, Stefanie and Konidaris, George},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2019},
  month={November}
}
```
