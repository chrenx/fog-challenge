## How to run

The program is containerized by Apptainer. Since sif file is large and not able to upload to FDA site, we are going to build from .def file.
```
sudo apptainer build fog.sif fog_apptainer.def
```
```
sudo apptainer run --nv fog.sif
```
Before running the main program, make sure the test data are under [/test_data](test_data) folder, and the ground truth csv file goes to [/gt](gt) folder where there should be only one single file.

For each csv file under [/test_data](test_data), the filename should be one of the column name in ground truth csv file. For example, 'trial_1' in /test_data/trial_1.csv should be in the ground_truth_df.columns

Please **delete** existing files under [/test_data](test_data) and [/gt](gt) and **replace** with your validation data.

All generated files would go to [/submission](submission).

To reproduce the same statistics from our test data, please don't remove any files under [/self_gt](self_gt) and [/self_test_data](self_test_data), and run the following command:
```
python -m main --enable_self_test
```

You can run the program for FDA's validation data with following command:
```
python -m main
```

