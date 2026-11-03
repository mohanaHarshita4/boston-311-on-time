# Early Flag: Will a Boston 311 Request Go Overdue?

I built this project because I wanted to work with real city data and answer a question that actually matters to the people who use that data. When a Boston resident reports a pothole, a broken streetlight, or missed trash pickup, the city sets a target date for closing it. Some requests close on time and some do not, and nobody knows at the start which ones will get stuck. I wanted to see if I could predict that at intake, from information that is already known when the request comes in, so a likely late request could be flagged early instead of discovered late. That is why I call it Early Flag. The point is not to count late requests after they are already late. The point is to flag them before they get there.

## The question

Can I predict, at the time a request is opened, whether it will close overdue?

That question shaped every choice in this project. If a field is only known after closure, I did not use it, even if it would have made the results look better. A model that needs the answer to make its prediction is not useful in real work.

## Data, and how I checked it first

The data comes from the City of Boston, Analyze Boston, 311 Service Requests, at https://data.boston.gov/dataset/311-service-requests

I downloaded the yearly files with src/download_data.py. I used the 2025 file for the analysis and the models. The raw files are stored locally in data/raw and they are not committed to GitHub, because the 2025 file alone is about 153 MB.

The 2025 file has 276,093 rows. Of those, 205,871 are closed and 70,222 are open. Among closed requests, 88.85 percent are on time and 11.15 percent are overdue.

Before I modelled anything, I checked the timing fields carefully, because the whole project depends on dates being right. The timestamps in the file are not in one format. Some have milliseconds and some have whole seconds. When I first parsed them with the default setting, about a third of the rows were dropped silently. I caught that by counting rows before and after parsing, and I fixed it by parsing with format mixed, which handles both formats. After that fix, open date and closed date parsed with no missing values, and target date was missing for about 8.9 percent of closed requests. Those rows have no target date, so I cannot compute a target window for them, and I dropped them from modelling instead of guessing a window.

That checking step matters to me. A prediction built on badly parsed dates would be wrong in a quiet way, and no model choice later would fix it.

## What the data showed before modelling

I explored the data with src/eda.py. The full summary is in reports/eda_summary.md, with figures in reports/figures.

Most requests close fast. The median time to close is 0.28 days, but the mean is 8.26 days. That gap tells me a small number of requests take a very long time. The median target window, target date minus open date, is 2.88 days.

Overdue risk is very different by request type, and that was the most useful finding from exploration. Recycling Cart Return went overdue 91.6 percent of the time, out of 4,511 requests. Sign Repair went overdue 69.9 percent of the time, out of 4,713. Request for Pothole Repair went overdue 43.5 percent of the time, out of 5,931. By contrast, Improper Storage of Trash (Barrels) went overdue only 0.2 percent of the time, and Needle Pickup went overdue 0.0 percent of the time. A single average of 11.15 percent overdue hides all of that, which is why request type became a core feature.

The largest departments for closed requests were PWDx with 124,034, BTDT with 32,301, and ISD with 15,648. The most common sources were Employee Generated with 89,571, Citizens Connect App with 55,484, and Constituent Call with 29,761. The largest neighbourhoods were Dorchester with 26,715, Roxbury with 20,629, and South End with 17,250.

## Target, and what I refused to use

Target: is_overdue is 1 if on_time is OVERDUE, else 0. I train only on closed requests, because open requests do not have an outcome yet.

I used only fields known at intake: case_title, subject, reason, type, department, neighborhood, source, open hour, open weekday, open month, and the target window in days, which is target date minus open date.

I did not use closed_dt, closure_reason, case_status, or on_time itself as a feature. All of those exist only after closure, and using them would leak the answer into the model.

## Features

Features are built in src/features.py.

1. Categorical fields are turned into numbers by one hot encoding. In plain terms, a value like Dorchester becomes its own column, with 1 for rows from Dorchester and 0 for other rows. I do the same for request type, department, source, and the other categorical fields.
2. Categories seen fewer than 500 times in the training set are grouped as Other. Without that step, I had 1,745 one hot columns, most of them almost always zero, and logistic regression was too slow to finish. Grouping rare values keeps the common types, like Recycling Cart Return and Sign Repair, as their own columns, and keeps the model practical.
3. Numeric fields are the target window in days, open hour, open weekday, and open month.
4. Splits are built in memory and used directly by the model scripts. I do not save the one hot tables as CSV files, because writing that wide table as CSV was also too slow.

## How I split train and test

I did not do a random split. I split by time.

1. Train: January to September 2025, 148,675 closed requests, 12.16 percent overdue.
2. Test: October to December 2025, 38,869 closed requests, 12.54 percent overdue.

In real work you train on the past and predict the future. A random split would mix later requests into training and let the model learn from months it should not have seen yet. The time split is stricter, and it is the honest test for this question.

In total I used 187,544 closed requests. The rest of the 205,871 closed requests were dropped because they had no target date.

## Models, and how I judged them

All models use the same features and the same time split. I judged them on accuracy, on F1 for the overdue class, and on recall for the overdue class, not on accuracy alone.

Accuracy alone is misleading here. Only about 12.5 percent of the test set is overdue. If I always predict on time, I get 87.46 percent accuracy and I catch zero overdue requests. That looks good on paper and does nothing for the goal, which is to flag likely late requests early. Recall tells me, out of all requests that actually went overdue, how many I caught. F1 balances precision and recall, so a model cannot win just by flagging everything.

1. Majority baseline, always predict on time. This is not a real model. It is the bar every other model has to beat.
2. Logistic regression, scikit-learn. It learns a weight for each feature, adds them up, and turns the sum into a probability of going overdue. If that probability is above 0.5, it predicts overdue.
3. Linear SVM, scikit-learn, LinearSVC. It tries to find a straight boundary that separates overdue from on time requests as cleanly as possible. I used a linear SVM, not an RBF kernel SVM, because an RBF kernel on about 149,000 training rows is far too slow. Linear keeps the comparison with logistic fair, because both are linear models on the same features.
4. Small MLP in PyTorch. This is the next model to train. It can learn combinations, like type plus neighbourhood plus target window together, in a way a linear model cannot. I have not trained it yet, so I am not claiming results for it here.

## Results

Test set: October to December 2025, 38,869 requests, 12.54 percent overdue.

| Model | Accuracy | F1 (overdue) | Overdue recall |
|---|---|---|---|
| Majority (always ONTIME) | 0.8746 | 0.0000 | 0.0000 |
| Logistic regression | 0.9104 | 0.5676 | 0.4687 |
| SVM (linear) | 0.9102 | 0.5648 | 0.4644 |

Reading this plainly: majority gets 87.46 percent accuracy by never predicting overdue, and it catches none of the overdue requests. Logistic regression beats it on accuracy and catches about 47 out of every 100 overdue requests, with an F1 of 0.5676. Linear SVM is essentially tied with logistic, slightly behind on all three measures. The differences are small, 0.0002 on accuracy, 0.0028 on F1, and 0.0043 on recall, which makes sense because both are linear models on the same features.

Detailed results are in reports/baseline_results.md and reports/svm_results.md.

## Errors, and what they taught me

The most costly error here is a missed overdue request, meaning it actually went overdue but the model predicted on time. A missed request does not get flagged early, which defeats the purpose of the project.

src/error_analysis.py retrains the logistic baseline and breaks missed requests down by request type and by neighbourhood. It only shows groups with at least 20 overdue requests in the test set, and it sorts by number missed, not by recall alone. A group with 2 overdue requests, both missed, would show zero recall but it is only 2 requests. A group with hundreds of missed requests is a bigger problem, even if its recall looks higher. Results are in reports/error_analysis.md after running the script.

## What I learned from building this

Most of the work was not in picking a fancy model. It was in getting the data right, in checking dates, in refusing to leak the answer, and in splitting by time so the test actually meant something. The first logistic fit was too slow because of too many rare columns, and the first file I wrote for features had an indentation problem that broke the whole function. I fixed both by simplifying, grouping rare values, building splits in memory, and replacing the whole file with a checked version instead of patching it line by line. Those fixes are part of the work, and I would rather say that plainly than pretend the first version worked.

## How to run

1. Install dependencies: pip install -r requirements.txt
2. Download the data: python src/download_data.py
3. Explore the data: python src/eda.py
4. Check the features: python src/features.py
5. Run the baselines: python src/baselines.py
6. Run the linear SVM: python src/svm.py
7. Run the error check: python src/error_analysis.py

Download data puts the 2024 and 2025 CSV files in data/raw. Exploration writes the summary and figures to reports. Features prints train and test sizes. Baselines, SVM, and error check each print a results table and write it to reports.

## Code

The code lives in src. Download data is in download_data.py. Exploration is in eda.py. Features are in features.py. Majority and logistic are in baselines.py. Linear SVM is in svm.py. Error check is in error_analysis.py. Reports are in reports, with eda_summary.md, baseline_results.md, svm_results.md, error_analysis.md, and figures.

## Author

Mohana Harshita Rangubhotla

MS Computer Science student at Khoury College of Computer Science, Northeastern University. I am taking Machine Learning this semester, and I built this project to put that coursework into practical use on a real city dataset.
