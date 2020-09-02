# Toward Quantifying Ambiguities in Artistic Images

This repository contains the source code and dataset of the visual indeterminacy project. 
More information can be found on the [Project page](http://cybertron.cg.tu-berlin.de/xiwang/tap-project-page/).

![teaser](http://cybertron.cg.tu-berlin.de/xiwang/tap-project-page/files/teaser-02.jpg)


### Citation 

If you use the data or the code, please consider citing the following paper. 
```
@article{,
author = {Wang, Xi and Bylinskii, Zoya and Hertzmann, Aaron and Pepperell, Robert},
title = {Towards Quantifying Ambiguities in Artistic Images},
year = {2020},
issue_date = {September 2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
journal = {ACM Trans. Appl. Percept.},
month = sep,
}
```

## Setup

* Install dependencies using pip/anaconda/etc, if needed: Streamlit, ...

* Download image set from `https://drive.google.com/file/d/1jKCcRIECSJSLO6M7GYGJyukKetpGh-Ox/view?usp=sharing` and unzip in main folder
* Optionally, download precomputed from 
[link](https://drive.google.com/file/d/1EnI10uc3UlagtTtx4DLIVaBEmoJkQMe7/view?usp=sharing). 

## To run:

To run the app: 
```
cd src
streamlit run main.py
```

### Interface
You can change selections in the sidebar on the left to see different visualizations.

* `Image Category`: select "All" to see all the images.

- `Display mode`: a drop-down menu with the following four options. 
    - `Single image`: shows details of each image in the selected category.     
        - `Show descriptions`: prints out all raw descriptions gathered from participants.
        - `Show descriptions table`: shows a table view of how descriptions are parsed to nouns, from which synonyms are further grouped together.
        - `Plot histograms of nouns`: By default, histograms of nouns from descriptions in both viewing condition (0.5s and 3s) is plotted.                
    - `First N images`: shows the first N images from the selected category.
    - `Scatterplot`: `X axis` and `Y axis` can be selected from the drop-down menus.  
    Fig. 6 in the published paper can viewed by selecting `H_3000` for `X axis` and `H_500` for `Y axis`.
    Mouseover a dot to see the name of that image.
    - `Sorted images`: displays images sorted by different metrics, including entropy from both viewing conditions,
     entropy difference, and entropy from 3s viewing.

### License 

Copyright (c) 2020 Xi Wang, Zoya Bylinskii, Aaron Hertzmann and Adobe Research

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

