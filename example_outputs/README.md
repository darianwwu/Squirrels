# Example Outputs

This folder contains example output files of the RFID workflow to illustrate the expected result structure and formats.

## Files

| File | Description |
|---|---|
| `segments_kept.csv` | Tabular list of all filtered motion segments including timestamps, RFID assignment, confidence scores, and nut events per segment. |
| `segments_kept.json` | The same segment data in JSON format, supplemented with video metadata and time calibration models. |
| `summary_per_squirrel.csv` | Summary per squirrel (RFID label) with total time spent, number of visits, nut counts, and first/last sighting timestamps. |
| `overlay_kept_segments.mp4` | Example video with overlaid display of detected segments, RFID labels, and nut events. |
