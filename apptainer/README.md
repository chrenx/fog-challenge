## How to run

The program is containerized by Apptainer. Since sif file is large and not able to upload to FDA site, we are going to build from .def file.
```
sudo apptainer build fog.sif fog_apptainer.def
```
```
sudo apptainer run --nv fog.sif
```
Before running the main program, make sure the test data are under [/test_data](test_data) folder, and the ground truth csv file goes to [/gt](gt) folder where there should be only one single file.

The test data under [/test_data](test_data) should follow the name format: (e.g.) **Trial60.csv** so that they can match the name in ground truth file.

Please **delete** existing files under [/test_data](test_data) and [/gt](gt) and **replace** with your validation data.

Now you can run the program:
```
python -m main
```

