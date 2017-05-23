# text_objseg_coco
## Usage  
1. `cd refer`  
2. `make`  
3. download refcoco+ expressions and annotations from http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip  
4. `mkdir data` and extract `refcoco+.zip` into `data`  
5. download and extract `coco_bbox.zip` into `refer`  
6. download refcoco+ image dataset (3.3G - this may take a long time)  
`python download_dataset.py`  
7. extract vocabularies  
`python extract_vocabulary.py`  
8. extract data from refcoco+  
`python parse_data.py`  
9. extract bounding box proposal  
`python parse_bbox.py`  
10. `cd ..` and train the model according to ronghang/text_objseg, only change path `exp-referit` into `refer`

