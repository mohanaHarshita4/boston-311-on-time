# EDA Summary — Boston 311, 2025

Source: City of Boston, Analyze Boston, 311 Service Requests.
File: `data/raw/boston311_2025.csv` (downloaded Oct 29, 2026).

## Counts
- Rows in 2025 file: 276,093
- Closed requests: 205,871
- Open requests: 70,222

## Target
- ONTIME among closed: 88.85%
- OVERDUE among closed: 11.15%
- This is imbalanced, so later models will be judged on F1 and recall for OVERDUE, not accuracy alone.

## Timing
- Median days to close: 0.28 days
- Mean days to close: 8.26 days (long tail)
- Median SLA window (target - open): 2.88 days
- Open date range (closed): 2025-01-01 to 2025-12-31

## Most common closed request types
case_title
Improper Storage of Trash (Barrels)    20007
Requests for Street Cleaning           19135
CE Collection                          18560
Parking Enforcement                    15013
Needle Pickup                          11636
Poor Conditions of Property             8835
Pick up Dead Animal                     8240
Unshoveled Sidewalk                     6383
Request for Pothole Repair              5931
Rodent Activity                         5159

## Overdue rate by most common request type
                                         n  overdue_rate
case_title                                              
Improper Storage of Trash (Barrels)  20007      0.002149
Requests for Street Cleaning         19135      0.044839
CE Collection                        18560      0.006519
Parking Enforcement                  15013      0.048092
Needle Pickup                        11636      0.000000
Poor Conditions of Property           8835      0.003848
Pick up Dead Animal                   8240      0.027670
Unshoveled Sidewalk                   6383      0.011593
Request for Pothole Repair            5931      0.435340
Rodent Activity                       5159      0.002908
Sign Repair                           4713      0.699130
Recycling Cart Return                 4511      0.916427

Takeaway for modelling: overdue risk is not uniform. For example, in this
snapshot Recycling Cart Return and Sign Repair go overdue far more often
than trash or needle pickup. That is why request type is a core intake feature,
and why a model that learns combinations (type + neighbourhood + SLA window)
can beat a single average.

## Top departments, sources, neighbourhoods (closed)
Departments:
department
PWDx    124034
BTDT     32301
ISD      15648
PARK     12346
GEN      11724
INFO      5184
PROP      2853
BWSC      1642

Sources:
source
Employee Generated      89571
Citizens Connect App    55484
Constituent Call        29761
City Worker App         27770
Self Service             3283
Maximo Integration          2

Neighbourhoods:
neighborhood
Dorchester                                26715
Roxbury                                   20629
South End                                 17250
Allston / Brighton                        16524
South Boston / South Boston Waterfront    14148
East Boston                               12486
Jamaica Plain                             12069
Back Bay                                  11700

## Leakage rule
Use only intake-known fields: case_title, subject, reason, type, department,
neighborhood, source, open hour/weekday/month, SLA window.
Never use closed_dt, closure_reason, case_status, or on_time as a feature.
