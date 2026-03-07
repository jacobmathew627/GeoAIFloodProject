# CHAPTER 1: INTRODUCTION

## 1.1 Background
Floods are increasingly frequent and severe disasters, occurring when natural or artificial drainage capacities are overwhelmed by intense precipitation, river overflows, or storm surges. In rapidly developing urban landscapes, traditional management strategies—often reliant on sparse ground observations and historical manual surveys—frequently fail to capture the high-resolution spatial dynamics of inundation. 

In Kerala, the 2018 and 2019 flood events exposed critical gaps in real-time disaster preparedness. The increasing availability of high-resolution geospatial datasets and satellite imagery presents a new frontier for improving flood risk analysis. By transitioning from reactive responses to proactive predictive modeling, disaster management can significantly reduce socioeconomic losses.

### 1.1.1 Need for Disaster Management in Ernakulam
Disaster management in flood-prone districts like Ernakulam is a proactive necessity. It involves the systematic identification of high-risk zones and the implementation of mitigation strategies before a crisis occurs. Modern frameworks leverage computational models to support early warnings and tactical evacuation planning.

### 1.1.2 Flood Scenario in Ernakulam, Kerala
Ernakulam’s unique physiography, featuring a complex network of coastal lowlands, dense river systems, and rapid urbanization, makes it uniquely vulnerable. The district faces high susceptibility during extreme monsoon events, where wetland encroachment and insufficient drainage infrastructure exacerbate the risk.

### 1.1.3 Role of Geospatial Technologies 
Geographic Information Systems (GIS) and Remote Sensing (RS) are the backbones of modern flood analysis. In this project, GIS facilitates the integration of multispectral spatial datasets, while Remote Sensing (Sentinel-1 SAR) provides all-weather surface monitoring capabilities. These tools allow for the creation of high-resolution Digital Elevation Models (DEM) and Land Use Land Cover (LULC) maps used to identify flood triggers.

### 1.1.4 Emergence of Geospatial Artificial Intelligence (GeoAI)
GeoAI integrates advanced Deep Learning techniques with spatial data. By training models on non-linear interactions between terrain (slope, elevation), hydrology (flow accumulation), and land cover, GeoAI can predict susceptibility with higher accuracy than conventional hydraulic models. This project utilizes a **Hybrid UNet-based CNN architecture** for spatial susceptibility mapping, which dynamically assesses risk based on variable rainfall inputs.

## 1.2 Problem Definition
Existing systems in Kerala often operate in isolation, lacking the fusion of real-time satellite radar data with dynamic rainfall scenarios. Many models are static and fail to provide responsive risk alerts based on live meteorological trends. 

The problem addressed in this project is the design of an integrated **GeoAI system** that fuses diverse spatial datasets (DEM, SAR, LULC) with **Live Weather API Data (Open-Meteo)** to generate dynamic flood susceptibility maps and visualize them through an interactive **Streamlit dashboard**.

## 1.3 Objectives
1. **Data Integration**: Collect and preprocess high-resolution geospatial datasets (Copernicus DEM, Sentinel-1 SAR, LULC) for the Ernakulam region.
2. **Meteorological Integration**: Integrate the **Open-Meteo API** to ingest live precipitation data and support scenario-based rainfall sensitivity analysis.
3. **CNN Susceptibility Mapping**: Implement a **UNet (CNN) model** to generate multi-channel flood susceptibility maps (Standard, Robust, and Supercharged modes).
4. **Hybrid Risk Fusion**: Calculate a **Flood Risk Index (FRI)** by fusing AI-predicted probabilities with physical hydrological constraints (Flow Accumulation, Slope).
5. **Interactive Visualization**: Develop a **Streamlit web dashboard** for real-time visualization, coordinate-aware risk inspection, and situational alerts.

## 1.4 Scope of the Project
The project is focused on **Ernakulam District, Kerala**, utilizing a high-resolution 30m grid. The scope includes:
*   Multi-modal inference: Standard (Topography), Robust (SAR-Fusion), and Supercharged (Full GeoAI).
*   Deep Learning implementations using the UNet architecture for spatial susceptibility.
*   Automated raster alignment and tile-based inference for scalable analysis.
*   Integration with live weather forecasting services for dynamic risk assessment.

## 1.5 Methodology Overview
1.  **Preprocessing**: Alignment of DEM, SAR, and LULC rasters to a common grid.
2.  **Rainfall Ingestion**: Fetching live precipitation from the **Open-Meteo API** or defining manual rainfall scenarios (e.g., 100mm, 150mm, 200mm).
3.  **GeoAI Inference**: Executing UNet-based spatial scanning to identify vulnerability patterns.
4.  **Risk Calculation**: Computing FRI through weighted fusion of AI results and physical terrain proxies (Flow, Slope, Elevation).
5.  **Dashboard Deployment**: Visualizing results via Folium-integrated web UI with hotspot identification and alert logic.
