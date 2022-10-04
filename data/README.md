The data folder is a placeholder for where data files would be saved when running the code in this repository. This prevents the user from needing to create a new folder when starting and ensures that file paths will match so long as a user places the data in the correct folder locations.  

Folder descriptions:

<details>
<summary>crop_yield</summary> 

The `crop_yield` folder contains the district level boundary crop yield data.
</details>

<details>
<summary>geo_boundaries</summary> 

The `geo_boundaries` folder contains the country boundary geomeotries as well as the district level boundary geometries.
</details>

<details>
<summary>land_cover</summary> 

The `land_cover` folder contains the worldwide cropland coverage rasters, the same rasters cropped to a country level, and the dataframes of points at the 0.01 degree resolution and the associated cropland percentages.
</details>

<details>
<summary>random_features</summary> 

The `random_features` folder contains the dataframes of points at the 0.01 degree resolution and the associated random convolutional features extracted from the imagery. the subfolders correspond to which satellite or satellite collection the features were extraced from. 
</details>