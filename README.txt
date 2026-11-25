README – Indonesian Earthquake Catalog (BMKG)

1. Dataset Title
Indonesian Earthquake Catalog (BMKG), 1998–2024

2. Description
This dataset contains the official earthquake catalog compiled and maintained
by the Badan Meteorologi, Klimatologi, and Geofisika (BMKG).

3. Data Coverage
Spatial coverage: Indonesian region
Temporal coverage: 1998–2024
Coordinate system: WGS84
Time standard: UTC

4. File Description
File name: BMKG_Earthquake_Catalog.csv
Format: CSV

5. Column Description
- Event ID: Unique BMKG event identifier (BMKG-YYYYMMDDhhmmss-NNN)
- Date: Origin date (UTC)
- Time (UTC): Origin time (UTC)
- Latitude: Epicenter latitude (decimal degrees)
- Longitude: Epicenter longitude (decimal degrees)
- Depth (km): Hypocentral depth in kilometers
- Magnitude: Earthquake magnitude
- Mag Type: Magnitude type (e.g., mb, ML, Mw, - if not specified)
- Source: Processing center (NC or RC)
- Source Event ID: Original event identifier from the source center (if available)

6. Event ID Definition
Each earthquake event is assigned a unique BMKG Event ID following the format
BMKG-YYYYMMDDhhmmss-NNN, where NNN distinguishes events occurring within the same second.

7. National Center (NC) and Regional Centers (RC)

Earthquake events in this catalog are processed by the BMKG National Center (NC) and ten Regional Centers (RC). 

Regional Centers (RC) are operational units of BMKG responsible for defined geographical regions. The RC codes are used in this dataset solely to indicate the processing responsibility and are not tied to administrative city boundaries.

In general, the National Center is responsible for events of larger magnitude and events requiring national-level processing, while Regional Centers handle routine seismic events within their respective regions.

Although Regional Centers primarily process smaller-magnitude events, the National Center may also process events of magnitude below 5.0, particularly when higher-quality solutions are available.

8. Event Selection and Priority Rule

In cases where the same earthquake event is reported by both the National Center and a Regional Center, the National Center solution is retained in the final catalog.

This priority rule is applied because all National Center events undergo centralized quality control procedures, regardless of magnitude, ensuring higher consistency and reliability of the event parameters.

9. License
Creative Commons Attribution 4.0 International (CC BY 4.0)

10. Contact
Badan Meteorologi, Klimatologi, and Geofisika (BMKG)







