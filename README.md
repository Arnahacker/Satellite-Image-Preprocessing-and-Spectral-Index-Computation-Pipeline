#  Satellite Image Preprocessing and Spectral Index Computation Pipeline

###  Overview
This project performs **radiometric**, **flat-field**, and **atmospheric corrections** on raw satellite imagery using **GDAL** and the **Py6S atmospheric radiative transfer model**.  
After correction, it computes key **spectral indices** (NDVI, NDWI, SAVI, MNDWI, NDBI) to identify and classify surface features such as vegetation, water, soil, and urban areas.  
The workflow produces scientifically corrected, analysis-ready images suitable for downstream machine learning or GIS tasks.

---
##  Processing Stages

###  Radiometric Correction
Removes **sensor noise and dark current** using a dark frame image.  
```python
corrected_band = band_sat - band_dark
```

###  Flat-Field Correction
Normalizes pixel brightness variations caused by **sensor or illumination inconsistencies**.
```python
normalized_band = corrected_band / band_flat
```

###  Atmospheric Correction (6S Model)
Applies **atmospheric correction** using the **Py6S radiative transfer model** to convert TOA (Top-of-Atmosphere) reflectance into **surface reflectance**.
```python
s = SixS()
s.atmos_corr = AtmosCorr.AtmosCorrLambertian
s.aero_profile = AeroProfile.Continental
s.run()
```

###  Spectral Index Computation
Calculates key indices:
| Index | Formula | Purpose |
|--------|----------|----------|
| **NDVI** | (NIR - Red) / (NIR + Red) | Vegetation health |
| **NDWI** | (Green - NIR) / (Green + NIR) | Water bodies |
| **SAVI** | (NIR - Red) / (NIR + Red + 0.5) × 1.5 | Soil-vegetation contrast |
| **MNDWI** | (Green - SWIR) / (Green + SWIR) | Improved water detection |
| **NDBI** | (SWIR - NIR) / (SWIR + NIR) | Urban and built-up areas |

---

##  Output Files
| File | Description |
|------|--------------|
| `corrected_dark.tif` | After dark-frame subtraction |
| `corrected_flat.tif` | After flat-field normalization |
| `corrected_atmospheric.tif` | After 6S atmospheric correction |
| *Index plots* | NDVI, NDWI, SAVI, MNDWI, NDBI + Masks (Vegetation, Water, Soil, Urban) |


---
