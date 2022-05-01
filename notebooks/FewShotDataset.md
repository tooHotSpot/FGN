
First & second datasets define the base and novel groups of categories selection rules

Class selection properties:
- 3/4 of total categories are selected as base, 1/4 of total categories are selected as novel

Selection properties (for few-shot-training stage):
- S-train-1: instances for already predefined base categories are selected from the base dataset
- S-train-2: the base dataset may contain novel categories instances, but those instances are deleted  

Selection properties (for few-shot-finetuning stage):
- S-finetune-1: instances for already predefined novel categories are selected from the novel dataset
- S-finetune-2: (3K * base + K * novel)-rule - knowledge transition stage will not suffer overfitting
- S-finetune-3: any-rare annotations for novel categories in the base dataset are considered  
- S-finetune-4: any-rare annotations for base categories in the novel dataset are considered
- S-finetune-5: delete images from the base dataset with more than 3K instances for any base class 
- S-finetune-6: delete images from the novel dataset with more than K instances for any novel class 
- S-finetune-7: if it is hard to collect data for finetuning, a novel class instance is cut from a new image
- S-finetune-8: images may contain single class instance but harder images with several categories depicted preferred
- S-finetune-9: in each episode, N categories selection is performed from both base and novel categories pool 
- S-finetune-10: by the article, instances for base categories are selected from the base dataset, but on practice
selection may be performed differently to diminish an overfitting / not training effect. In a cross-dataset setting,
- S-finetune-4 may not be useful (COCO2VOC, OMNIISEG2MNISTISEG), in single-dataset setting S-finetune-4 is vital       
                                  
Selection properties (for few-shot-test stage):
- S-test-1: delete from novel dataset images selected on the finetuning stage
- S-test-2: images with no single instance for any novel class are deleted
- S-test-3: during the testing (by FGN rule), for an every novel class on a single image, N - 1 novel categories 
are selected randomly, while support set is build from the annotations selected on the finetuning stage
- S-test-4: base categories instances are deleted

1)  Single-dataset setting like VOC2VOC (or simpler OMNI2OMNI, MNIST2MNIST, COCO2COCO)
    - Subsets for base dataset and novel dataset have to be different (e.g. train or val splits), because
    it is hard to retrain on images which have appeared during the training stage several hundreds of times
    - E.g. VOC2VOC with 20 categories: 15 random are novel, other 5 are base 
               
2) Cross-dataset setting like COCO2VOC (or more harder VOC2COCO)
    - E.g. VOC2COCO is too hard, not considering it, main problem is the COCO dataset complexity
    - E.g. COCO2VOC is more real, and FGN authors experiment on transition from COCO-train to VOC-val: 
    60 categories from COCO are base, 20 categories overlapped with VOC are novel   
    - Annotations for base categories in the novel dataset should be considered in production but ignored in academic task  
    - Since COCO contains the novel categories the one thing left is to choose 
    an insufficient amount of images for an every novel class from the VOC-val 
    
3) Cross-dataset setting list OMNIISEG2MNISTISEG
    - The especial detail in this setting is that the object are not presented 
    in any way compared to the single-dataset setting where we do not see this 
    detail immediately until just thinking about the finetuning data merge and preparation
    - The setting is more close to real cases, especially for applications with top 
    cross-the-globe novelty with objects not depicted in any dataset 
    - OMNIISEG2MNISTISEG is just the test setting used to open the environment
    - Development and experimentation are not guided to solve the setting but may be considered in the future 
