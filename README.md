# text_objseg_coco
## Usage  
1. `cd refer && make`  
2. download refcoco+ expressions and annotations from http://tlberg.cs.unc.edu/licheng/referit/data/refcoco+.zip  
3. `mkdir data` and extract `refcoco+.zip` into `data`  
4. download and extract `coco_bbox.zip` into `refer`  
5. download refcoco+ image dataset (3.3G - this may take a long time)  
`python download_dataset.py`  
6. extract vocabularies  
`python extract_vocabulary.py`  
7. extract data from refcoco+  
`python parse_data.py`  
8. extract bounding box proposal  
`python parse_bbox.py`  
9. `cd ..` and train the model according to ronghang/text_objseg, only change path `exp-referit` into `refer`

