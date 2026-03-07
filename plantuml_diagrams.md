# GeoAI Flood Project - Architecture & Flow Diagrams

This document contains PlantUML code for visualizing the project's logic and architecture, strictly following the system workflow using the actual **Hybrid CNN (UNet)** architecture and **Live Weather API** integration.

## 1. Refined Activity Diagram
This diagram shows the complete workflow from data ingestion to final flood risk visualization and hotspot generation, accurately reflecting the CNN-based processing and API-driven meteorology.

```plantuml
@startuml
skinparam backgroundColor #fdfdfd
skinparam activity {
    BackgroundColor #E1F5FE
    BorderColor #01579B
    ArrowColor #01579B
    FontName "Verdana"
    FontSize 13
}

start

:User Opens Dashboard;
:Uploads or Updates Data Sources;

if (Data Valid?) then (Yes)

    partition "Meteorological Integration" {
        if (Use Live Weather?) then (Yes)
            :Fetch Precipitation from Open-Meteo API;
        else (No)
            :Select Rainfall Scenario (100-200mm);
        endif
        :Preprocess Rainfall and Raster Data;
    }

    partition "AI Susceptibility Module" {
        :Load Pre-trained UNet (CNN) Model;
        :Predict Flood Susceptibility;
        note left: Multi-channel (DEM, SAR, LULC)
    }

    partition "GIS Terrain Module" {
        :Perform GIS Terrain Analysis;
        :Compute Drainage Density, Slope, Elevation;
    }

    partition "Risk Fusion Engine" {
        :Fuse AI Susceptibility and GIS Risk Layers;
        :Calculate Flood Risk Index (FRI);
        note right: FRI = Max(AI_Prob, Phys_Risk)
    }

    partition "Output Generation" {
        :Identify Hotspot Regions;
        :Generate GeoTIFF & GeoJSON Outputs;
    }

    :Display Flood Risk Map and Alerts;

else (No)

    :Show Error Message to User;

endif

stop
@enduml
```

## 2. Sequence Diagram
Highlights the interaction between the User, Dashboard, AI Engine (CNN), and GIS Processor.

```plantuml
@startuml
actor User
participant "Dashboard UI" as UI
participant "Open-Meteo API" as API
participant "AI Module (UNet CNN)" as AI
participant "GIS Terrain Engine" as GIS
database "Data Storage" as DB

User -> UI : Opens Dashboard & Uploads Data
UI -> UI : Validate Data Source
alt Data Valid
    alt Live Mode
        UI -> API : Fetch Current Precipitation
        API --> UI : rainfall_mm
    else Manual Mode
        User -> UI : Set Rainfall Slider
    end
    
    UI -> AI : Request Flood Susceptibility
    AI -> DB : Fetch & Preprocess Rasters (DEM/SAR/LULC)
    AI -> AI : Run CNN Inference (UNet)
    AI --> UI : AI Susceptibility Layer
    
    UI -> GIS : Request Terrain Analysis
    GIS -> GIS : Compute Slope/Elevation/Drainage
    GIS --> UI : GIS Risk Layer
    
    UI -> UI : Fuse AI & GIS Layers (FRI Calculation)
    UI -> UI : Identify Hotspots & Risk Levels
    UI -> DB : Generate GeoTIFF/GeoJSON
    UI -> User : Update Folium Map & Alerts
else Data Invalid
    UI -> User : Show Error Message
end
@enduml
```

## 3. Component Diagram
Structural view of the system's modular architecture using the UNet CNN engine and Open-Meteo integration.

```plantuml
@startuml
package "GeoAI Dashboard" {
    [UI Controller] as UI
    [Map Renderer (Folium)] as Map
    [Result Analytics] as Analytics
}

package "AI Engine (UNet CNN)" {
    [UNet Architecture] as UNet
    [Tile Processor] as Tile
    [Feature Normalizer] as Norm
}

package "Meteorological Service" {
    [Open-Meteo API Client] as Met
}

package "GIS Processor" {
    [Terrain Analyzer]
    [Hydro Formulas] as Hydro
}

folder "Data Management" {
    [Input Tiffs (DEM/SAR/LULC)] as Inputs
    [Final Risk Outputs] as RO
}

UI --> Met : Fetch Live Rain
UI --> Norm : Send Data
Tile --> UNet : Batch Processing
UNet --> RO : Save Probabilities
Hydro --> RO : Physics Overlays
Inputs --> Norm : Data Stream
Map --> RO : Visualize Results
@enduml
```
