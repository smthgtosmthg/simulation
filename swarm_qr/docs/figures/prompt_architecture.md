# Prompt — architecture of the proposed system

This text describes the system to be represented: its components, what each one holds, what
travels between them, and the technology behind each block. It prescribes content only.
Layout, colours and graphical style are left entirely to whoever draws the figure.

---

## The system to describe

A cooperative system of N aerial vehicles that takes a QR-code inventory of a warehouse it
has never seen before. There is no pre-computed path: the vehicles decide where to go from
what they have observed so far. Three properties have to come across: the team shares a
single memory, each vehicle decides for itself from that memory, and no commitment is made
before something has actually been observed.

Everything runs in **Isaac Sim 5.1** with the **Pegasus Simulator** (Iris airframes) and one
**ArduPilot SITL** instance per vehicle, driven over **MAVLink** (`pymavlink`). The physics
step is 1/800 s, and the autopilots are in the loop: the system commands velocities, it does
not teleport anything.

## The environment, which the system does not control

- racks and cartons are placed differently in every building, from a procedural generator
- labels are real QR codes glued on the faces of the cartons, on both sides of an aisle
- an obstacle may appear while the mission is running
- a vehicle may be lost in flight
- the system learns about all of this only through its sensors

## Components

### The aerial vehicle — one of N, identical, running on board

**Sensors.** Two lateral cameras at 1024 × 768 with a 60° field of view, used for reading.
One forward camera at 160 × 120. A 3D LiDAR doing a full revolution, 360° horizontally and
36° vertically, useful between 0.4 m and 25 m. The vehicle pose, supplied by the autopilot.
All of these are the native Isaac Sim sensors.

**Perception.**
- *Learned spotter*: **YOLO11n** (Ultralytics), 1024 px input, half precision on GPU, two
  classes — `qr` and `carton`, confidence threshold 0.5. It finds cartons and labels at
  range, without decoding them.
- *Decoder*: **zxing-cpp** by default, with **ZBar** (`pyzbar`) and OpenCV's
  `QRCodeDetector` available as alternatives. It reads a label only from close range.
- *3D placement*: the LiDAR range measured along the ray that passes through the detection,
  plus `cv2.solvePnP` for the orientation of the label, since the physical size of the code
  is known.
- *Volume*: the LiDAR rays separate free volume from solid volume.

**Flight controller.** Transit along the path it was given. A slow approach, then holding
in front of the face to be read. Abandonment when the path is cut or when progress stops.
It emits velocity setpoints in `MAV_FRAME_LOCAL_NED` to the autopilot in GUIDED mode; the
autopilot does the attitude control.

### The shared map — one single instance, common to the whole team

A 3D grid in `numpy`, 25 cm cells, holding four layers:

- **Occupancy**: solid, free, or not yet observed.
- **Oriented coverage**: what has been observed and from which direction, on four bits — a
  face seen from the wrong side does not count as covered.
- **Label table**: each label goes from unknown, to spotted but unread, to read; with its
  position, its orientation and its identifier. The same code can appear on two faces.
- **Teammates and claims**: where the others are, and which targets they have claimed;
  a claim expires unless it is renewed.

Paths are found with **A\*** over a cost grid derived from the occupancy layer, where a cell
that has never been observed carries a cost instead of being forbidden.

### The decision — run by each vehicle, once per cycle

- **Candidate generation**: read a spotted label, cover a face never observed, or explore;
  all three are written the same way, as a pose and a viewing direction.
- **Scoring on a single scale**: plus the expected return, minus the path cost, minus a
  penalty if the target is already claimed, minus a penalty if a teammate is close, plus
  the semantic advice multiplied by its weight λ.
- **Feasibility and selection**: keep only what can be reached through space known to be
  free, best candidate first.

### The semantic guide — optional, weight λ

A small open vision-language model loaded from Hugging Face through
`AutoModelForImageTextToText`: **SmolVLM-500M-Instruct**, **SmolVLM-Instruct (2B)**, or a
**Qwen2-VL** class model, quantised to 4-bit NF4 with `bitsandbytes`, fp16 compute, SDPA
attention, and an optional **LoRA** adapter through `peft`. It is shown the side camera and
the map seen from above, with the candidate zones numbered, and asked two questions: which
zone to visit next, and from which side to approach the rack there. It runs asynchronously,
and λ = 0 removes it from the system without breaking anything. It is a side input, not a
link in the main loop.

### Mission supervision — outside the vehicles

- **Rhythms**: flight control, decision and advice each run at their own rate.
- **Progress**: distinct identifiers read, against the expected inventory.
- **Termination**: a share of the inventory reached, or no new identifier for a delay.

## What travels between the components

Each exchange has to be named, not merely suggested by a line between two boxes.

| From | To | What travels |
|---|---|---|
| the warehouse | the sensors | light and geometry, the only channel into the system |
| perception | shared map | decoded codes, spotted labels, free and solid volume |
| shared map | candidate generation | the current state of knowledge |
| shared map | scoring | existing claims and teammate positions |
| semantic guide | decision | an advised area, weighted by λ |
| shared map | semantic guide | a summary of the map, as an image seen from above |
| flight controller | shared map | the claim released when the target is abandoned |
| selection | shared map | the claim on the chosen target, expiring unless renewed |
| selection | flight controller | the chosen pose and the path to it |
| shared map | supervision | mission progress |

Two of these exchanges close the loop: the decision writes back into the map, and the
vehicle's perception feeds the map again. This closure is the point of the figure — the
system has no beginning and no end, only a cycle.

## The loop, in five steps

1. The vehicles observe, and write into the map.
2. The map holds what is now worth doing.
3. Each vehicle prices the available actions and claims one.
4. The controller flies it, or gives it up.
5. What it establishes goes back into the map.

## The principle to convey

Nothing is written before it is observed, and nothing is claimed for longer than it is
renewed. These two rules are what cover a warehouse never seen before, an obstacle that
appears mid-mission, and the loss of a vehicle — no special case is added for any of them.

## What must not be left out

- there are N identical vehicles, but only one map
- perception writes to the map, decision reads from it; they never talk to each other
- the guide is removable, and its removal changes nothing else
- the three rhythms are different: flight control, decision, advice
- a claim expires by itself, which is what makes the loss of a vehicle harmless
- the autopilot sits below the flight controller, and is not part of the contribution

## The stack, in one table

| Role | Technology |
|---|---|
| Simulator and physics | Isaac Sim 5.1, physics step 1/800 s |
| Airframes and bridge | Pegasus Simulator, Iris |
| Autopilot | ArduPilot SITL, one instance per vehicle, GUIDED mode |
| Link to the autopilot | MAVLink, `pymavlink` |
| Cameras | Isaac Sim native, 2 × 1024 × 768 lateral at 60°, 1 × 160 × 120 forward |
| LiDAR | Isaac Sim RTX range sensor, 360° × 36°, 0.4 – 25 m |
| Spotting cartons and labels | YOLO11n, Ultralytics, 1024 px, fp16 |
| Decoding QR codes | zxing-cpp; ZBar and OpenCV as alternatives |
| Label pose | LiDAR range along the ray, `cv2.solvePnP` |
| Map and paths | `numpy` grid at 25 cm, A\* over the cost grid |
| Semantic guide | SmolVLM-500M / SmolVLM-2B / Qwen2-VL, 4-bit NF4, LoRA optional |
