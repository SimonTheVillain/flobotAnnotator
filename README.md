# Dirt Annotator
This tool is used to annotate planar regions in video sequences. 
Given a rosbag containing image and TF data, annotations are propagated between frames.

To counter insufficiencies in tracking the annotated regions are allowed to be shifted at keyframes.
Positions in between are interpolated.


## ROSBAG Files
### Topics
### TF Frames

## Parameters
TODO: implement these
* --input the input rosbag for annotation
* --output either a rosbag (ending with .bag) or a folder with images (png) and masks.
* --annotation_path Path to the yaml file storing the annotations.
* --use_depth use the depth values to fit a ground plane. If not the x,y plane is assumed.


//TODO: add functionality to change topic names

## Controls
Since this tool was used to only annotate a few small
* `esc` Aborting process.
* `space` Pause at current frame.
* `a` go back one frame.
* `d` go forward one frame.
* `return` accept input
* `delete` ... does what it says
* `p` store annotation file
* `e` end time of selection (IMPLEMENTED?)
* `start` time of selection (IMPLEMENTED?)


## Usage

Run the simulation by specifying a input ros file:

`BLABLABLABLABLA`

On a given frame press `space` to freeze the playback and use the cursor together with left click to draw annotations.

Pressing `return` finalizes the polygon whereas `esc` aborts the creation process.

Pressing `space` again will continue playback. It is alternatively possible to step trough single images by pressing
 `a` and `d`.


It is possible to edit markers at keyframes. By clicking a annotations at the given 
keyframes we can edit single datapoints of of polygons by clicking and dragging.

Dragging the whole polygon is done by either dragging at its center or, if it is too small, 
dragging while pressing the `shift` key.  


## Flobot
The FLOBOT - (Floor Washing Robot for Professional Users) project among other goals aimed to reach intelligent cleaning behaviour by utilizing **dirt detection**.

The project itself did not result in a powerful DNN based solution due to a lack of proper datasets but it resulted in one online learning approach.

The Gaussion Mixture Model (GMM) based approach originating from this project approached the problem as novelty detection. 
All it takes is one frame to learn the GMM, which is only descriptive enough to describe the floor and discard dirt as outlier.
## Dirt Dataset
To evaluate dirt detection approaches and, if we eventually end up with enough data, train DNN based approaches we need datasets.

TODO: Describe the dataset. link to the files and so on.

## Aknowledgement

This work is supported by the European Comission through the Horizon 2020 Programme (H2020-ICT-2014-1, Grant agreement no.~645376), FLOBOT.


