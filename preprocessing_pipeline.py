from osgeo import gdal
import numpy as np
from Py6S import SixS, AtmosCorr, AeroProfile
import rasterio
import matplotlib.pyplot as plt

#radiometric correction
sat_img = gdal.Open("input.tif")
dark_frame = gdal.Open("dark_frame.tif")
band_sat = sat_img.GetRasterBand(1).ReadAsArray()
band_dark = dark_frame.GetRasterBand(1).ReadAsArray()
corrected_band = band_sat - band_dark
driver = gdal.GetDriverByName("GTiff")
driver = gdal.GetDriverByName("GTiff")
output = driver.Create("corrected_dark.tif", sat_img.RasterXSize, sat_img.RasterYSize, 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(corrected_band)
output.SetProjection(sat_img.GetProjection())
output.SetGeoTransform(sat_img.GetGeoTransform())
output = None

#flatfield correction
flat_field = gdal.Open("flat_field.tif")
band_flat = flat_field.GetRasterBand(1).ReadAsArray().astype(np.float32)
sat_img = gdal.Open("corrected_dark.tif")
corrected_band = sat_img.GetRasterBand(1).ReadAsArray().astype(np.float32)
band_flat = np.where(band_flat == 0, 1e-6, band_flat)
normalized_band = corrected_band / band_flat
driver = gdal.GetDriverByName("GTiff")
output = driver.Create("corrected_flat.tif", sat_img.RasterXSize, sat_img.RasterYSize, 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(normalized_band)
output.FlushCache()
output = None
sat_img = None
flat_field = None

#Converting TOA Reflectance to Surface Reflectance (Using 6S Model)

from Py6S import SixS, AtmosCorr, AeroProfile
import numpy as np

s = SixS()
s.atmos_corr = AtmosCorr.AtmosCorrLambertian
s.aero_profile = AeroProfile.Continental

s.run()

toa_reflectance = (normalized_band * s.outputs.direct_solar_irradiance) / s.outputs.global_gas_transmittance

toa_reflectance = np.nan_to_num(toa_reflectance, nan=0.0)

driver = gdal.GetDriverByName("GTiff")
output = driver.Create("corrected_atmospheric.tif", sat_img.RasterXSize, sat_img.RasterYSize, 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(toa_reflectance)
output.FlushCache()
output = None

#didn't understand geometric

def compute_indices(image_path, bands):
    with rasterio.open(image_path) as src:
        nir = src.read(bands['NIR']).astype(np.float32)
        red = src.read(bands['Red']).astype(np.float32)
        green = src.read(bands['Green']).astype(np.float32)
        swir = src.read(bands['SWIR']).astype(np.float32)

    np.seterr(divide='ignore', invalid='ignore')

    small_value=pow(10,10^-8)
    ndvi = (nir - red) / (nir + red+small_value)
    ndwi = (green - nir) / (green + nir)
    savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
    mndwi = (green - swir) / (green + swir)
    ndbi = (swir - nir) / (swir + nir)

    return {"NDVI": ndvi, "NDWI": ndwi, "SAVI": savi, "MNDWI": mndwi, "NDBI": ndbi}

  ans= compute_indices(image_path, bands)


def generate_masks(indfrom osgeo import gdal
import numpy as np
from Py6S import SixS, AtmosCorr, AeroProfile
import rasterio
import matplotlib.pyplot as plt

# Radiometric Correction
sat_img = gdal.Open("input.tif")
dark_frame = gdal.Open("dark_frame.tif")
band_sat = sat_img.GetRasterBand(1).ReadAsArray()
band_dark = dark_frame.GetRasterBand(1).ReadAsArray()
corrected_band = band_sat - band_dark
driver = gdal.GetDriverByName("GTiff")
output = driver.Create("corrected_dark.tif", sat_img.RasterXSize, sat_img.RasterYSize, 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(corrected_band)
output.SetProjection(sat_img.GetProjection())
output.SetGeoTransform(sat_img.GetGeoTransform())
output = None

# Flat-Field Correction
flat_field = gdal.Open("flat_field.tif")
band_flat = flat_field.GetRasterBand(1).ReadAsArray().astype(np.float32)
sat_img = gdal.Open("corrected_dark.tif")
corrected_band = sat_img.GetRasterBand(1).ReadAsArray().astype(np.float32)
band_flat = np.where(band_flat == 0, 1e-6, band_flat)
normalized_band = corrected_band / band_flat
output = gdal.GetDriverByName("GTiff").Create("corrected_flat.tif", sat_img.RasterXSize, sat_img.RasterYSize, 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(normalized_band)
output.FlushCache()
output = None
sat_img = None
flat_field = None

# Atmospheric Correction (6S)
s = SixS()
s.atmos_corr = AtmosCorr.AtmosCorrLambertian
s.aero_profile = AeroProfile.Continental
s.run()
toa_reflectance = (normalized_band * s.outputs.direct_solar_irradiance) / s.outputs.global_gas_transmittance
toa_reflectance = np.nan_to_num(toa_reflectance, nan=0.0)
driver = gdal.GetDriverByName("GTiff")
output = driver.Create("corrected_atmospheric.tif", normalized_band.shape[1], normalized_band.shape[0], 1, gdal.GDT_Float32)
output.GetRasterBand(1).WriteArray(toa_reflectance)
output.FlushCache()
output = None

# Geometric Correction (Reprojection)
input_image = "corrected_atmospheric.tif"
output_image = "geometrically_corrected.tif"
gdal.Warp(output_image, input_image, dstSRS="EPSG:4326", resampleAlg=gdal.GRA_Bilinear, xRes=30, yRes=30, outputType=gdal.GDT_Float32)

# Compute Spectral Indices
def compute_indices(image_path, bands):
    with rasterio.open(image_path) as src:
        nir = src.read(bands['NIR']).astype(np.float32)
        red = src.read(bands['Red']).astype(np.float32)
        green = src.read(bands['Green']).astype(np.float32)
        swir = src.read(bands['SWIR']).astype(np.float32)

    np.seterr(divide='ignore', invalid='ignore')
    small_value = 1e-8
    ndvi = (nir - red) / (nir + red + small_value)
    ndwi = (green - nir) / (green + nir + small_value)
    savi = ((nir - red) / (nir + red + 0.5)) * 1.5
    mndwi = (green - swir) / (green + swir + small_value)
    ndbi = (swir - nir) / (swir + nir + small_value)
    return {"NDVI": ndvi, "NDWI": ndwi, "SAVI": savi, "MNDWI": mndwi, "NDBI": ndbi}

bands = {"NIR": 4, "Red": 3, "Green": 2, "SWIR": 5}
indices = compute_indices("geometrically_corrected.tif", bands)

# Generate Masks
def generate_masks(indices):
    return {
        "Vegetation": indices["NDVI"] > 0.2,
        "Water": indices["MNDWI"] > 0,
        "Soil": (indices["SAVI"] > 0.1) & (indices["SAVI"] < 0.3),
        "Urban": indices["NDBI"] > 0
    }

# Plot Indices and Masks
def plot_indices_and_masks(indices, masks):
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for i, (name, index) in enumerate(indices.items()):
        ax[0, i].imshow(index, cmap='RdYlGn' if "NDVI" in name or "SAVI" in name else 'coolwarm')
        ax[0, i].set_title(name)
        ax[0, i].axis("off")
    for i, (name, mask) in enumerate(masks.items()):
        ax[1, i].imshow(mask, cmap='gray')
        ax[1, i].set_title(name + " Mask")
        ax[1, i].axis("off")
    plt.tight_layout()
    plt.show()

masks = generate_masks(indices)
plot_indices_and_masks(indices, masks)
