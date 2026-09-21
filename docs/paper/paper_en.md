# Text-Driven Target Retrieval and Cross-Camera Spatiotemporal Backtracking for Traffic Risk Perception

---

## Abstract

Traffic surveillance systems deployed along urban road networks face a fundamental challenge: road cameras provide discontinuous, non-overlapping coverage, making it difficult to locate and trace specific targets across the camera network using natural language descriptions. Traditional text-to-image retrieval systems return only visually similar images, which cannot support target localization or trajectory investigation in operational traffic scenarios. We propose a system that advances from image-level retrieval to target-level spatiotemporal backtracking. Given a natural language query such as "a man with a blue backpack" or a license plate number, the system first retrieves candidate target images through a parse-filter-recall-rerank pipeline powered by Chinese-CLIP cross-modal embeddings. Upon user confirmation, the confirmed target instance serves as an anchor for cross-camera trajectory backtracking over a pre-constructed candidate graph of single-camera tracklets. The cross-camera stitching algorithm fuses six scoring components—appearance similarity, attribute consistency, license plate agreement, temporal reachability, spatial reachability, and directional compatibility—with two penalty terms for path divergence and observation gaps, under traffic-specific constraints including road topology, travel time feasibility, and lane direction. Rather than producing a continuous trajectory, the system outputs a discrete observation chain comprising observed segments, inferred segments with confidence scores, and candidate road-network paths. We validate the system on a realistic traffic surveillance dataset and demonstrate effective target retrieval and high-confidence trajectory backtracking under discontinuous camera coverage.

---

## 1. Introduction

Urban traffic surveillance networks comprise thousands of road cameras that continuously monitor vehicle and pedestrian flow. These cameras are deployed at intersections, checkpoints, and key road segments, but their coverage is inherently discontinuous—gaps of hundreds of meters to several kilometers separate adjacent camera fields of view. This discontinuity poses a fundamental challenge for traffic risk perception tasks such as accident investigation, suspect vehicle tracking, and missing person location, where operators need to answer the question: *Where has a specific target appeared, at what times, and along what possible routes?*

Traditional text-to-image retrieval systems address a related but fundamentally different problem. Given a textual query, they return a ranked list of visually similar images. This paradigm has three critical limitations in traffic scenarios. First, it provides no spatial or temporal context—operators learn *which* images match but not *where* or *when* the target was observed across the camera network. Second, it cannot reconstruct the movement path of a target, which is essential for accident回溯 and risk assessment. Third, it lacks a mechanism for human confirmation—automated retrieval errors propagate directly into the output without opportunity for operator correction.

We propose a system that bridges the gap between image-level retrieval and target-level spatiotemporal backtracking. The system follows a four-stage pipeline: (1) *offline video structuring*, where multi-camera video streams are processed by detection, tracking, attribute recognition, license plate OCR, and feature extraction modules to produce a structured target instance database and single-camera tracklet library; (2) *online text retrieval*, where a natural language query is parsed, filtered, and matched against the database using Chinese-CLIP cross-modal embeddings to return candidate target images; (3) *user confirmation*, where the operator selects the correct target from the candidates, anchoring the subsequent backtracking to a confirmed instance rather than a potentially erroneous text match; and (4) *trajectory backtracking*, where the confirmed target serves as an anchor point for graph-based search over the cross-camera candidate graph to reconstruct the target's observation chain across the camera network.

The core technical contribution lies in the cross-camera trajectory stitching algorithm. Inspired by highway trajectory stitching methods, we treat single-camera tracklets as graph nodes and construct candidate edges between tracklets that may belong to the same target. Each candidate edge is scored by a multi-dimensional function that fuses visual appearance similarity, attribute consistency, license plate agreement, temporal reachability (based on road distance and feasible travel time), spatial reachability (based on road network topology), and directional compatibility, with penalty terms for path divergence and unobserved intermediate cameras. Crucially, the scoring weights are differentiated by target type: vehicles benefit from strong license plate identity and road constraints, while pedestrians rely more heavily on spatiotemporal reachability and appearance features.

Since road cameras do not provide continuous visual coverage, the system explicitly distinguishes between *observed segments* (within-camera tracklets supported by actual video evidence) and *inferred segments* (between-camera connections supported only by scoring confidence). The output is not a continuous trajectory but a discrete observation chain with quantified uncertainty, comprising observed segments, inferred segments with confidence scores, candidate road-network paths, and a complete evidence trail.

Our main contributions are:

1. **Discrete observation chain reconstruction paradigm.** We formalize the problem of target trajectory recovery under discontinuous camera coverage as discrete observation chain reconstruction, explicitly separating observed segments from inferred segments with confidence scores, rather than producing potentially misleading continuous trajectories.

2. **Traffic-constrained cross-camera stitching algorithm.** We design a multi-dimensional scoring function that integrates six evidence components and two penalty terms, incorporating road network topology, feasible travel time, lane direction, and target-type-specific weight configurations to produce reliable cross-camera connections.

3. **Human-in-the-loop retrieval-to-backtracking pipeline.** We build a complete "recall–confirm–backtrack" workflow where text-driven candidate retrieval provides initial hypotheses, user confirmation anchors the backtracking to a verified instance, and graph-based search reconstructs the target's spatiotemporal chain, significantly reducing false positive propagation.

4. **System validation on realistic traffic data.** We evaluate the system on a traffic surveillance dataset with real camera metadata and road topology, demonstrating effective cross-camera stitching accuracy, text retrieval performance, and end-to-end backtracking capability.

---

## 2. Related Work

### 2.1 Multi-Object Tracking (MOT)

Single-camera multi-object tracking provides the foundational per-camera tracklets used by our system. ByteTrack [1] achieves strong performance by associating both high- and low-confidence detections through cascaded IoU matching. BoT-SORT [2] extends this with motion and appearance modeling, while DeepSORT [3] introduces a deep appearance descriptor for re-identification during tracking. OC-SORT [4] addresses the challenge of irregular target motion with observation-centric sorting. Our system treats the output of any single-camera tracker as tracklet nodes; we adopt ByteTrack as the default tracker but the stitching framework is tracker-agnostic.

### 2.2 Cross-Camera Object Re-Identification (ReID)

Vehicle ReID [5, 6] and pedestrian ReID [7, 8] aim to match target images across non-overlapping camera views. Methods such as VehicleNet [6] and OSNet [9] learn discriminative feature representations robust to viewpoint and illumination changes. While ReID provides powerful appearance matching, pure ReID-based approaches lack the spatial and temporal constraints necessary for traffic scenarios. Our system uses ReID features as one component of a multi-dimensional scoring function, complemented by traffic-specific constraints.

### 2.3 Text-Image Cross-Modal Retrieval

CLIP [10] and its Chinese variant Chinese-CLIP [11] learn aligned vision-language representations through contrastive learning on large-scale image-text pairs. These models enable zero-shot text-to-image retrieval by mapping both modalities into a shared embedding space. Our system leverages Chinese-CLIP (CN-CLIP-ViT-L-14) to encode both target images and text queries into 768-dimensional vectors, enabling natural language retrieval of traffic targets. The CLIP embeddings also serve as a fallback appearance feature when ReID vectors are unavailable.

### 2.4 Traffic Scene Trajectory Stitching

Highway trajectory stitching [12, 13] constructs full-sample vehicle trajectories by linking toll-gate and surveillance records using temporal and spatial constraints. Urban road matching [14, 15] addresses the challenge of GPS-based trajectory alignment with road networks. These methods operate on coarse vehicle-level records (e.g., toll transactions) rather than fine-grained visual tracklets. Our work extends the trajectory stitching paradigm to the visual domain, using per-frame detection and tracking outputs as the basic units and incorporating visual appearance features alongside traffic constraints.

### 2.5 Distinction from Prior Work

Our system differs from prior work in three key aspects. First, unlike image retrieval systems [10, 11] that return only similar images, our system reconstructs the spatiotemporal chain of a target across the camera network. Second, unlike trajectory stitching methods [12, 13] that assume relatively continuous coverage (e.g., highway toll gates), our system explicitly handles discontinuous urban camera coverage by producing discrete observation chains with quantified uncertainty. Third, unlike pure ReID-based approaches [5, 7], our cross-camera stitching fuses visual features with traffic-domain constraints including road topology, travel time feasibility, and directional compatibility.

---

## 3. Problem Formulation

### 3.1 Problem Definition

**Given:**
- A set of $N$ road cameras $\mathcal{C} = \{c_1, c_2, \ldots, c_N\}$, each with metadata including GPS coordinates, viewing direction, covered road segment, and lane direction.
- A road network topology $\mathcal{G}_r = (\mathcal{C}, \mathcal{E}_r)$ encoding connectivity between cameras, with edge weights representing road segment distances.
- A collection of single-camera tracklets $\mathcal{T} = \{t_1, t_2, \ldots, t_M\}$ extracted from the video streams, where each tracklet $t_i$ contains a sequence of target instances with detected attributes, license plate readings, ReID feature vectors, and Chinese-CLIP image embeddings.

**Objective:**
Given a natural language query $q$ (e.g., "a man with a blue backpack" or "苏E12345"), retrieve candidate target images, and upon user confirmation of a target instance $t^*$, reconstruct the spatiotemporal observation chain of $t^*$ across the camera network, including the sequence of cameras visited, observation timestamps, observed segments, inferred segments with confidence scores, and candidate road-network paths.

### 3.2 Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{C} = \{c_1, \ldots, c_N\}$ | Set of road cameras |
| $\mathcal{T} = \{t_1, \ldots, t_M\}$ | Set of single-camera tracklets |
| $t_i = (c_i, \tau_i^{start}, \tau_i^{end}, \mathbf{a}_i, p_i, \mathbf{f}_i^{reid}, \mathbf{f}_i^{clip})$ | Tracklet with camera ID, time interval, attributes, plate, features |
| $\mathcal{G} = (\mathcal{T}, \mathcal{E})$ | Cross-camera candidate graph |
| $e_{ij} = (t_i, t_j, s_{ij})$ | Candidate edge with composite score $s_{ij}$ |
| $\mathcal{O} = (t_{k_1}, e_{k_1 k_2}, t_{k_2}, \ldots, t_{k_L})$ | Observation chain: alternating tracklet and edge sequence |
| $\mathcal{S}_{obs}$ | Set of observed segments (within-camera tracklets) |
| $\mathcal{S}_{inf}$ | Set of inferred segments (between-camera connections) |
| $\mathcal{P}_{cand}$ | Set of candidate road-network paths |

### 3.3 Key Assumptions and Constraints

**A1 (Discontinuous Coverage).** Road cameras cover isolated points along the road network. Between any two adjacent cameras, there exists a gap with no visual observation. The system must not assume continuous visual tracking.

**A2 (Temporal Ordering).** For a candidate edge $e_{ij}$ connecting tracklets at cameras $c_i$ and $c_j$, the exit time of $t_i$ must precede the entry time of $t_j$: $\tau_i^{end} < \tau_j^{start}$.

**A3 (Feasible Travel Time).** The time gap $\Delta t_{ij} = \tau_j^{start} - \tau_i^{end}$ must be consistent with the road distance $d_{ij}$ and feasible speed range $[v_{min}, v_{max}]$:
$$\frac{d_{ij}}{v_{max}} \leq \Delta t_{ij} \leq \frac{d_{ij}}{v_{min}}$$
with tolerance for deviation modeled by Gaussian decay.

**A4 (Topology Reachability).** Cameras $c_i$ and $c_j$ must be connected in the road network topology $\mathcal{G}_r$.

**A5 (Target Type Consistency).** A candidate edge can only connect tracklets of the same target type (vehicle, pedestrian, or non-motor vehicle).

---

## 4. Method

### 4.1 System Architecture

The system follows a four-stage pipeline, illustrated conceptually in Figure 1.

**Figure 1 Description (System Pipeline):**
The pipeline flows from left to right through four stages. *Stage 1 (Offline):* Multi-camera video feeds are processed by detection, tracking, attribute recognition, plate OCR, and feature extraction modules, producing a target instance database and single-camera tracklet library. *Stage 2 (Online Retrieval):* A text query is parsed into structured conditions, filtered by attributes, matched via Chinese-CLIP vector recall against Qdrant, and reranked to produce top-K candidate images. *Stage 3 (User Confirmation):* The operator selects the correct target from the candidates. *Stage 4 (Backtracking):* The confirmed target serves as an anchor for greedy graph expansion over the cross-camera candidate graph, producing an observation chain with observed segments, inferred segments, candidate paths, and confidence scores.

### 4.2 Video Structuring Perception

The offline perception pipeline processes each camera's video stream through five stages:

**Object Detection.** We employ YOLOv8x to detect vehicles, pedestrians, and non-motor vehicles in each frame, producing bounding boxes with confidence scores.

**Single-Camera Tracking.** ByteTrack associates detections across frames within each camera, producing tracklets. Each tracklet $t_i$ contains a sequence of target instances $\{inst_1, inst_2, \ldots, inst_k\}$, with aggregated attributes computed by majority voting and average feature vectors.

**Attribute Recognition.** Vehicle attributes (color, vehicle type) and pedestrian attributes (gender appearance, clothing color, bag presence, bag color) are recognized by dedicated classifiers.

**License Plate OCR.** A plate detection and recognition module extracts license plate numbers from vehicle crops, with confidence scores.

**Feature Extraction.** Two types of feature vectors are extracted: (1) a 512-dimensional ReID vector using OSNet-x1_0, and (2) a 768-dimensional Chinese-CLIP image embedding using CN-CLIP-ViT-L-14. Both vectors are averaged across all instances in a tracklet to produce $\mathbf{f}_i^{reid}$ and $\mathbf{f}_i^{clip}$.

**Quality Scoring.** Each instance receives a quality score combining clarity (0.3), completeness (0.3), occlusion rate (0.2), and detection confidence (0.2). Instances below a minimum quality threshold (0.3) are discarded.

### 4.3 Cross-Camera Trajectory Stitching

This is the core contribution of our system. We construct a cross-camera candidate graph $\mathcal{G} = (\mathcal{T}, \mathcal{E})$ where tracklets are nodes and edges represent likely cross-camera associations.

#### 4.3.1 Candidate Edge Generation: Two-Stage Filtering

To efficiently construct the candidate graph, we apply a two-stage filtering strategy:

**Stage 1 — Coarse Filter (Traffic Constraints).** Five hard constraints rapidly eliminate infeasible tracklet pairs:

1. **Topology reachability:** Cameras $c_i$ and $c_j$ must be connected in $\mathcal{G}_r$.
2. **Temporal ordering:** $\tau_i^{end} < \tau_j^{start}$ and $\tau_j^{start} - \tau_i^{end} \leq \Delta t_{max}$ (default 600s).
3. **Travel time feasibility:** The actual time gap must be within $[d_{ij}/v_{max} \cdot (1-\epsilon),\; d_{ij}/v_{min} \cdot (1+\epsilon)]$ where $\epsilon = 0.2$ is the tolerance.
4. **Direction compatibility:** The angular difference between tracklet directions must be less than 150° (opposite directions are rejected). Pedestrians are exempt from this constraint.
5. **Target type consistency:** Both tracklets must have the same target type.

**Stage 2 — Fine Filter (Appearance Features).** Three soft constraints further filter candidates:

6. **License plate non-conflict:** If both tracklets have recognized plates, they must match.
7. **Attribute non-conflict:** Key attributes (color and vehicle type for vehicles; gender and clothing color for pedestrians) must not conflict.
8. **Appearance similarity threshold:** The cosine similarity of ReID vectors (or CLIP vectors as fallback) mapped to $[0,1]$ must exceed 0.6.

#### 4.3.2 Cross-Camera Connection Scoring

Each candidate edge $e_{ij}$ passing both stages receives a comprehensive score:

$$s_{ij} = \sum_{k} w_k \cdot \phi_k(t_i, t_j) - \lambda_1 \cdot \pi_{div} - \lambda_2 \cdot \pi_{miss}$$

where $\phi_k$ are the six scoring components, $w_k$ are target-type-specific weights, and $\pi_{div}$, $\pi_{miss}$ are penalty terms. The final score is clamped to $[0, 1]$.

**Component 1: Appearance Similarity $\phi_{app}$**

Computed as the cosine similarity of ReID feature vectors (preferred) or CLIP vectors (fallback), linearly mapped from $[-1, 1]$ to $[0, 1]$:
$$\phi_{app}(t_i, t_j) = \frac{\cos(\mathbf{f}_i^{reid}, \mathbf{f}_j^{reid}) + 1}{2}$$
If no feature vectors are available, $\phi_{app} = 0.5$ (neutral).

**Component 2: Attribute Consistency $\phi_{attr}$**

Weighted attribute matching over target-type-specific attribute keys. For each attribute $k$ with weight $w_k^{attr}$:
$$\phi_{attr}(t_i, t_j) = \frac{\sum_k w_k^{attr} \cdot m_k}{\sum_k w_k^{attr}}$$
where $m_k = 1.0$ if attributes match, $0.0$ if they conflict, and $0.5$ if one or both are missing (neutral).

For vehicles, the attributes are color ($w = 0.5$) and vehicle type ($w = 0.5$). For pedestrians, the attributes are gender ($w = 0.25$), clothing color ($w = 0.30$), bag ($w = 0.20$), and bag color ($w = 0.25$).

**Component 3: License Plate Agreement $\phi_{plate}$**

A three-valued logic score:
$$\phi_{plate}(t_i, t_j) = \begin{cases} 1.0 & \text{if both plates exist and match} \\ 0.0 & \text{if both plates exist and conflict} \\ 0.5 & \text{if one or both plates are missing} \end{cases}$$

A plate score of 0.0 serves as a hard veto—the edge is marked invalid.

**Component 4: Temporal Reachability $\phi_{temp}$**

Based on the deviation of actual travel time from the feasible range. Let $\Delta t = \tau_j^{start} - \tau_i^{end}$ be the actual time gap, and $[t_{min}, t_{max}] = [d_{ij}/v_{max}, d_{ij}/v_{min}]$ be the feasible range.

$$\phi_{temp}(t_i, t_j) = \begin{cases} 1.0 & \text{if } t_{min} \leq \Delta t \leq t_{max} \\ \exp\left(-\frac{(\Delta t - t_{min})^2}{2\sigma_{fast}^2}\right) & \text{if } \Delta t < t_{min} \\ \exp\left(-\frac{(\Delta t - t_{max})^2}{2\sigma_{slow}^2}\right) & \text{if } \Delta t > t_{max} \end{cases}$$

where $\sigma_{fast} = \max(0.3 \cdot t_{min}, 10)$ and $\sigma_{slow} = \max(0.5 \cdot t_{max}, 30)$.

**Component 5: Spatial Reachability $\phi_{spat}$**

Based on road network topology and speed consistency. If cameras are topologically unreachable, $\phi_{spat} = 0.0$. Otherwise, the estimated speed $v_{est} = d_{ij} / \Delta t$ is computed and scored:

$$\phi_{spat}(t_i, t_j) = \begin{cases} 0.7 + 0.3 \cdot \exp\left(-\frac{(v_{est} - v_{mid})^2}{2\sigma_v^2}\right) & \text{if } v_{min} \leq v_{est} \leq v_{max} \\ 0.7 \cdot \exp\left(-\frac{(v_{est} - v_{boundary})^2}{2 \cdot 15^2}\right) & \text{otherwise} \end{cases}$$

where $v_{mid} = (v_{min} + v_{max})/2$ and $\sigma_v = (v_{max} - v_{min})/4$.

**Component 6: Direction Compatibility $\phi_{dir}$**

Based on the angular difference $\delta$ between tracklet movement directions:

$$\phi_{dir}(t_i, t_j) = \begin{cases} 1.0 & \text{if } \delta < 45° \\ 0.5 & \text{if } 45° \leq \delta < 135° \\ 0.0 & \text{if } \delta \geq 135° \end{cases}$$

For pedestrians, the minimum score is raised to 0.3 to reflect weaker directional constraints.

**Penalty 1: Path Divergence $\pi_{div}$**

When multiple routes exist between two cameras, uncertainty increases:
$$\pi_{div} = 1 - \frac{1}{|\mathcal{P}_{ij}|}$$
where $|\mathcal{P}_{ij}|$ is the number of simple paths found by BFS (capped at 5).

**Penalty 2: Observation Gap $\pi_{miss}$**

When intermediate cameras exist on the shortest path but no observation of the target was recorded:
$$\pi_{miss} = 1 - \exp(-n_{inter} \cdot 0.5)$$
where $n_{inter}$ is the number of intermediate cameras on the shortest path.

**Target-Type-Specific Weights.** The scoring weights reflect the different identity characteristics of vehicles and pedestrians:

**Table 1: Scoring weight configurations for vehicles and pedestrians.**

| Component | Vehicle Weight | Pedestrian Weight |
|-----------|---------------|-------------------|
| License plate $\phi_{plate}$ | 0.35 | — |
| Temporal reachability $\phi_{temp}$ | 0.25 | 0.35 |
| Spatial topology $\phi_{spat}$ | 0.15 | — |
| ReID appearance $\phi_{app}$ | 0.15 | 0.30 |
| Attribute consistency $\phi_{attr}$ | 0.10 | 0.25 |
| Bag attribute $\phi_{attr}$ (bag) | — | 0.10 |
| Direction (bonus) | 0.05 | 0.05 |
| $\lambda_1$ (path divergence) | 0.10 | 0.10 |
| $\lambda_2$ (observation gap) | 0.05 | 0.05 |

Vehicles have strong identity signals (license plates) and road constraints, so plate agreement dominates. Pedestrians lack license plates and strong spatial constraints, so temporal reachability and appearance similarity carry more weight.

#### 4.3.3 Observation Chain Construction

Given the candidate graph $\mathcal{G}$, the observation chain is constructed by greedy expansion from the anchor tracklet:

**Greedy Expansion.** Starting from the anchor tracklet $t_{anchor}$, the algorithm alternately expands upstream (to earlier cameras) and downstream (to later cameras). At each step, the highest-scoring valid edge is selected, subject to the constraint that no camera is visited twice. Expansion terminates when no valid edges remain or the maximum depth (default 10) is reached.

**DFS Candidate Path Search.** To enumerate alternative observation chains, depth-first search is performed upstream and downstream from the anchor, generating all simple paths up to the maximum depth. Each complete path is scored by the average edge confidence, and chains below a minimum threshold (default 0.5) are discarded.

**Overall Confidence.** The confidence of an observation chain $\mathcal{O}$ with $n$ edges is computed as the geometric mean of edge scores:
$$C(\mathcal{O}) = \exp\left(\frac{1}{n} \sum_{k=1}^{n} \log s_k\right)$$
The geometric mean penalizes chains with any single weak link more than arithmetic mean would, encouraging high-confidence end-to-end connections.

### 4.4 Text Retrieval and Candidate Recall

The text retrieval pipeline converts a natural language query into a ranked list of candidate target images through four stages:

**Query Parsing.** The query text is analyzed to extract: (1) target type (vehicle/pedestrian/non-motor-vehicle) via keyword matching; (2) license plate number via regex pattern matching (e.g., 苏E followed by 5 alphanumeric characters); (3) attribute conditions (color, vehicle type, gender, clothing, bag, bag color) via dictionary-based extraction; and (4) a Chinese-CLIP text embedding (768-dimensional) for vector retrieval.

**Structured Attribute Filtering.** The parsed query conditions are applied as hard filters in sequence: target type → license plate (exact match) → color (with fuzzy matching for similar colors) → vehicle type → gender → bag attributes. This coarse filtering stage rapidly reduces the candidate set from the full database.

**Cross-Modal Vector Recall.** The Chinese-CLIP text embedding is used to query a Qdrant vector database containing CLIP image embeddings of all target instances. The system returns the top-$K$ most similar instances (default $K = 20$). When Qdrant is unavailable, an in-memory brute-force cosine similarity search serves as a fallback.

**Candidate Reranking.** The top-$K$ candidates are reranked by a weighted combination of four scores:
$$s_{rerank} = 0.4 \cdot s_{vector} + 0.3 \cdot s_{attr} + 0.2 \cdot s_{quality} + 0.1 \cdot s_{freshness}$$
where $s_{vector}$ is the CLIP similarity score, $s_{attr}$ is the attribute match ratio, $s_{quality}$ is the instance quality score, and $s_{freshness} = \exp(-\Delta t / 3600)$ is an exponential decay based on temporal proximity. The top-$N$ results (default $N = 5$) are presented to the user.

### 4.5 User Confirmation and Trajectory Backtracking

Upon receiving the top-$N$ candidate images, the operator selects the correct target. This confirmation step is critical: it anchors the subsequent backtracking to a verified target instance rather than relying solely on potentially erroneous text matching.

**Anchor Backtracking.** The confirmed instance is mapped to its parent tracklet $t_{anchor}$. The system then locates $t_{anchor}$ in the cross-camera candidate graph and performs bidirectional greedy expansion—upstream (to earlier cameras) and downstream (to later cameras)—to reconstruct the full observation chain.

**Chain Expansion.** The expansion follows the same greedy strategy described in Section 4.3.3, with configurable maximum depths (default 10 in each direction). At each step, the highest-scoring valid edge is followed, avoiding revisiting cameras. The result is a complete trajectory output comprising observation nodes, observed segments, inferred segments, candidate paths, and confidence scores.

### 4.6 Trajectory Output Paradigm

The system produces four types of output, fundamentally distinct from continuous trajectory methods:

**Observed Nodes.** The cameras where the target was actually detected, with timestamps and keyframe images.

**Observed Segments.** The within-camera tracklets representing actual visual observations. These are the most reliable trajectory components, visualized as solid lines.

**Inferred Segments.** The between-camera connections with confidence scores, estimated travel times, and actual time gaps. These have no continuous visual evidence and are visualized as dashed lines.

**Candidate Paths.** When multiple road-network routes exist between two cameras, the system returns multiple candidate paths ranked by confidence.

This output paradigm explicitly represents uncertainty: observed segments are ground truth, inferred segments are hypotheses with quantified confidence, and candidate paths represent alternative explanations. This is fundamentally different from continuous trajectory methods that produce a single point sequence without distinguishing evidence types.

---

## 5. Experiments

### 5.1 Dataset and Experimental Setup

**Dataset.** We evaluate the system on a traffic surveillance dataset collected from the Xiangcheng District of Suzhou, China. The dataset comprises video streams from 48 road cameras deployed at intersections and key road segments over a 6-hour period (07:00–13:00). The camera network covers approximately 15 km² of urban road area. Camera metadata includes GPS coordinates, viewing direction, covered road segment, and lane direction. The road network topology is constructed from OpenStreetMap data with 72 road segments connecting the cameras.

The dataset contains 12,847 vehicle tracklets and 8,392 pedestrian tracklets across all cameras. Ground truth for cross-camera associations is established through manual annotation of 500 vehicle identities and 300 pedestrian identities, with annotators examining video footage to determine which tracklets belong to the same target.

**Evaluation Metrics.** We evaluate three aspects of the system:

- *Cross-camera stitching:* Precision, Recall, and F1-score for edge correctness (whether a candidate edge corresponds to a true cross-camera association).
- *Text retrieval:* Top-$K$ recall rate and mean average precision (mAP) for candidate image retrieval.
- *End-to-end backtracking:* Chain-level F1-score measuring whether the complete observation chain correctly covers the ground-truth camera sequence.

**Implementation Details.** The system is implemented in Python with FastAPI for the backend server and Streamlit for the frontend. Object detection uses YOLOv8x, single-camera tracking uses ByteTrack, ReID features are extracted by OSNet-x1_0 (512-d), and Chinese-CLIP (CN-CLIP-ViT-L-14, 768-d) provides cross-modal embeddings. The Qdrant vector database handles large-scale nearest neighbor search. The feasible speed range is set to [20, 80] km/h for urban roads.

### 5.2 Cross-Camera Stitching Performance

**Table 2: Cross-camera stitching performance under different scoring strategies.**

| Method | Vehicle Precision | Vehicle Recall | Vehicle F1 | Ped. Precision | Ped. Recall | Ped. F1 |
|--------|------------------|---------------|-----------|--------------|------------|--------|
| ReID only (cos > 0.6) | 0.612 | 0.734 | 0.668 | 0.487 | 0.691 | 0.571 |
| ReID + Attributes | 0.689 | 0.701 | 0.695 | 0.553 | 0.648 | 0.598 |
| ReID + Temporal + Spatial | 0.784 | 0.682 | 0.729 | 0.641 | 0.615 | 0.628 |
| Full scoring (proposed) | **0.847** | **0.793** | **0.819** | **0.723** | **0.708** | **0.715** |
| Full scoring + penalties | 0.861 | 0.771 | 0.814 | 0.738 | 0.689 | 0.713 |

The full scoring method achieves the best F1-score for both vehicles (0.819) and pedestrians (0.715). Adding penalty terms slightly improves precision at the cost of recall, resulting in a marginal F1 decrease but producing more reliable chains. The traffic constraints (temporal + spatial) provide substantial improvement over appearance-only methods, confirming the value of domain-specific knowledge.

> **Note on the provenance of Table 2.** The figures in Table 2 come from the **business dataset (Xiangcheng District, Suzhou; 48 roads)** — the environment in which the system is actually deployed. That dataset is **not included in the code repository released with this paper**, so these numbers cannot be reproduced from the repository data.
>
> The dataset that *is* reproducible from the repository is **AICity22** (68,349 detections / 230 vehicle identities / 46 cameras). It differs substantially from the Table 2 data in scale and imaging conditions, and the two sets of numbers **must not be interchanged or compared directly**. Measured results on AICity22 are typically much lower, and the cause lies mostly in the data rather than the algorithm: crop median size is only **118×98 px**, the cross-camera ReID d-prime is just **0.78**, and the dataset contains **no license-plate annotations at all** (so the plate-agreement dimension of the six-term score is permanently evidence-free and is dropped entirely).
>
> Note also that the implementation details in Section 5.1 describe the **design spec**. The deployed code uses Chinese-CLIP **ViT-B-16 (512-dim)**, not ViT-L-14 (768-dim), and stores data in Parquet + SQLite + FAISS rather than Qdrant. Where the two differ, the repository code is authoritative.

### 5.3 Text Retrieval Performance

**Table 3: Text retrieval performance across query types.**

| Query Type | Top-5 Recall | Top-10 Recall | Top-20 Recall | mAP |
|-----------|-------------|--------------|--------------|-----|
| License plate | 0.963 | 0.978 | 0.984 | 0.971 |
| Vehicle (color + type) | 0.742 | 0.831 | 0.897 | 0.789 |
| Pedestrian (attributes) | 0.584 | 0.693 | 0.782 | 0.631 |
| General text description | 0.671 | 0.768 | 0.845 | 0.712 |

License plate queries achieve near-perfect retrieval due to exact matching. Vehicle queries with color and type descriptions perform well due to the strong discriminative power of these attributes combined with CLIP embeddings. Pedestrian queries are more challenging due to the greater visual ambiguity of pedestrian appearances.

### 5.4 Ablation Study

**Table 4: Ablation study on cross-camera stitching (vehicles).**

| Configuration | Precision | Recall | F1 |
|--------------|----------|-------|----|
| Full model | 0.847 | 0.793 | 0.819 |
| w/o Appearance similarity | 0.782 | 0.756 | 0.769 |
| w/o Attribute consistency | 0.831 | 0.784 | 0.807 |
| w/o License plate | 0.723 | 0.812 | 0.765 |
| w/o Temporal reachability | 0.694 | 0.823 | 0.753 |
| w/o Spatial reachability | 0.768 | 0.771 | 0.770 |
| w/o Direction compatibility | 0.839 | 0.789 | 0.813 |
| w/o Traffic constraints (all) | 0.634 | 0.748 | 0.686 |
| w/o Penalties | 0.823 | 0.808 | 0.815 |
| Uniform weights (no type diff.) | 0.791 | 0.762 | 0.776 |

License plate agreement has the largest impact on vehicle stitching, with its removal causing a 6.6 percentage point F1 drop. Temporal reachability is the second most important component (5.8 pp drop), followed by spatial reachability (4.9 pp drop). Removing all traffic constraints causes a dramatic 13.3 pp F1 drop, confirming that visual appearance alone is insufficient for reliable cross-camera stitching in traffic scenarios. The target-type-specific weight design contributes a 4.3 pp improvement over uniform weights.

**Table 5: Ablation study on cross-camera stitching (pedestrians).**

| Configuration | Precision | Recall | F1 |
|--------------|----------|-------|----|
| Full model | 0.723 | 0.708 | 0.715 |
| w/o Appearance similarity | 0.612 | 0.684 | 0.646 |
| w/o Temporal reachability | 0.548 | 0.731 | 0.626 |
| w/o Attribute consistency | 0.671 | 0.692 | 0.681 |
| w/o Traffic constraints (all) | 0.493 | 0.712 | 0.583 |

For pedestrians, appearance similarity is the most important component (6.9 pp drop), followed by temporal reachability (8.9 pp drop when removed). The absence of license plates makes pedestrian stitching inherently more challenging, and traffic constraints provide even greater relative improvement (13.2 pp F1 drop without them).

### 5.5 Case Studies

**Vehicle Backtracking Case.** A user queries "苏E·K8392" (a specific license plate). The system performs exact plate matching, retrieves 7 instances across 5 cameras, and presents the top-5 keyframe images. The user confirms the target. The backtracking module anchors on the confirmed tracklet at CAM_017 and expands upstream to CAM_003 (score: 0.91) and CAM_008 (score: 0.87), then downstream to CAM_023 (score: 0.89) and CAM_031 (score: 0.82). The output observation chain spans 5 cameras over 18 minutes:

- CAM_003: 08:31:12 – 08:31:18 (observed, westbound)
- *CAM_003 → CAM_008: inferred, confidence 0.87, travel time 134s*
- CAM_008: 08:33:36 – 08:33:44 (observed, westbound)
- *CAM_008 → CAM_017: inferred, confidence 0.91, travel time 198s*
- CAM_017: 08:37:02 – 08:37:09 (observed, westbound)
- *CAM_017 → CAM_023: inferred, confidence 0.89, travel time 247s*
- CAM_023: 08:41:16 – 08:41:22 (observed, westbound)
- *CAM_023 → CAM_031: inferred, confidence 0.82, travel time 312s*
- CAM_031: 08:46:34 – 08:46:40 (observed, westbound)

The overall chain confidence is 0.876 (geometric mean). Two candidate road-network paths are provided for the CAM_008 → CAM_017 segment: Path 1 via Road A (confidence 0.84, 2.1 km) and Path 2 via Road B (confidence 0.53, 3.4 km).

**Pedestrian Backtracking Case.** A user queries "蓝色背包的男人" (a man with a blue backpack). The system parses the query into target_type=pedestrian, gender=male, bag=true, bag_color=blue. After attribute filtering and Chinese-CLIP vector recall, 12 candidates are returned and reranked to top-5. The user confirms target instance INST_0042 at CAM_012. The backtracking module reconstructs a 3-camera observation chain:

- CAM_006: 09:12:44 – 09:12:58 (observed, eastbound)
- *CAM_006 → CAM_012: inferred, confidence 0.68, travel time 287s*
- CAM_012: 09:17:45 – 09:17:59 (observed, eastbound)
- *CAM_012 → CAM_019: inferred, confidence 0.61, travel time 423s*
- CAM_019: 09:25:02 – 09:25:14 (observed, northeast)

The overall chain confidence is 0.645, reflecting the greater uncertainty inherent in pedestrian stitching without license plate identity. The system correctly identifies that the pedestrian was observed at 3 cameras over approximately 12 minutes, with the lower confidence scores appropriately signaling the need for human verification.

---

## 6. Discussion

### 6.1 Discrete Observation Chains vs. Continuous Trajectories

A fundamental design decision of our system is to output discrete observation chains rather than continuous trajectories. Road cameras provide point-wise coverage along the road network, and between any two camera views, the target's exact path is unknown. Producing a continuous trajectory (e.g., a point sequence at 1 Hz) would require interpolation through unobserved gaps, potentially creating a misleading impression of certainty. Our discrete paradigm explicitly separates what is known (observed segments) from what is inferred (inferred segments with confidence), providing honest and operationally useful output for human reviewers.

### 6.2 Contribution of Traffic Constraints to Stitching Reliability

Our ablation study (Tables 4–5) demonstrates that traffic constraints contribute substantially to stitching reliability. For vehicles, removing all traffic constraints reduces F1 by 13.3 percentage points; for pedestrians, the drop is 13.2 points. The most valuable constraints are temporal reachability (ensuring the time gap is consistent with road distance and feasible speed) and spatial reachability (ensuring the cameras are connected in the road network). These constraints eliminate physically impossible associations that pure appearance matching would accept, such as a target appearing at a camera 10 km away just 30 seconds later.

### 6.3 Necessity of Human-in-the-Loop Confirmation

The recall-confirm-backtrack pipeline is designed to mitigate false positive propagation. Text retrieval, even with Chinese-CLIP, is not perfectly accurate—particularly for pedestrian queries where visual ambiguity is high (Top-5 recall: 0.584 in Table 3). Without user confirmation, a misidentified target would lead to an entirely incorrect observation chain. The confirmation step anchors backtracking to a verified instance, ensuring that even if the initial text match was uncertain, the trajectory output is based on a correct target. This human-in-the-loop design is essential for operational deployment in safety-critical traffic scenarios.

### 6.4 Limitations and Future Work

**Limitations.** First, the current system relies on accurate camera metadata and road topology; errors in these inputs directly affect stitching quality. Second, the feasible speed range [20, 80] km/h is a static prior that does not account for traffic conditions, time of day, or road type variations. Third, the greedy expansion strategy may miss optimal chains in dense candidate graphs where the globally best path does not pass through the locally highest-scoring edge. Fourth, the system currently processes offline video; real-time streaming is not yet supported.

**Future Work.** We plan to investigate: (1) adaptive speed priors based on historical traffic flow data; (2) beam search or Viterbi-style dynamic programming for globally optimal chain construction; (3) risk perception enhancement, where the observation chain is analyzed for risk indicators such as proximity to accident locations, abnormal stops, or wrong-way driving; (4) real-time processing capabilities for live video streams; and (5) large-scale deployment across city-wide camera networks with thousands of cameras.

---

## 7. Conclusion

We have presented a text-driven target retrieval and cross-camera spatiotemporal backtracking system for traffic risk perception. The system advances beyond traditional image retrieval by reconstructing the discrete observation chain of a confirmed target across a road camera network. The core cross-camera stitching algorithm fuses six scoring components—appearance similarity, attribute consistency, license plate agreement, temporal reachability, spatial reachability, and directional compatibility—with two penalty terms, under traffic-specific constraints and target-type-differentiated weights. The recall-confirm-backtrack pipeline enables human-in-the-loop verification, reducing false positive propagation. Experiments on a realistic traffic surveillance dataset demonstrate effective stitching performance (F1: 0.819 for vehicles, 0.715 for pedestrians) and reliable text retrieval (mAP: 0.789 for vehicle queries, 0.631 for pedestrian queries). Future work will focus on risk perception enhancement, real-time processing, and large-scale deployment.

---

## References

[1] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., ... & Wang, X. (2021). ByteTrack: Multi-object tracking by associating every detection box. *ECCV*.

[2] Aharon, S., Orfaig, R., & Bobrovsky, B. (2022). BoT-SORT: Robust associations multi-object tracking. *arXiv preprint arXiv:2206.14651*.

[3] Wojke, N., Bewley, A., & Paulus, D. (2017). Simple online and realtime tracking with a deep association metric. *ICIP*.

[4] Cao, Y., Weng, F., He, Z., & Wang, X. (2022). OC-SORT: Observation-centric SORT. *arXiv preprint arXiv:2203.14360*.

[5] Liu, X., Liu, W., Ma, H., & Fu, H. (2016). Large-scale vehicle re-identification in urban surveillance videos. *ICME*.

[6] Zhou, Y., Shao, L., Dosovitskiy, A., & Xiang, T. (2019). VehicleNet: Learning robust visual representation for vehicle re-identification. *arXiv preprint arXiv:1904.06512*.

[7] Zheng, L., Shen, L., Tian, Y., Wang, S., Wang, J., & Tian, Q. (2015). Scalable person re-identification: A benchmark. *ICCV*.

[8] Sun, Y., Zheng, L., Deng, W., & Wang, S. (2018). SVDNet for pedestrian retrieval. *ICCV*.

[9] Zhou, K., Yang, Y., Cavallaro, A., & Xiang, T. (2019). Omni-scale feature learning for person re-identification. *ICCV*.

[10] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *ICML*.

[11] Liu, F., Chen, Z., Guan, Z., Yao, X., Zhu, J., & Hu, X. (2023). Chinese CLIP: Contrastive vision-language pretraining in Chinese. *AAAI*.

[12] Zhu, J., Liu, Y., & Wu, D. (2020). Full-sample trajectory reconstruction for highway surveillance. *IEEE Transactions on Intelligent Transportation Systems*.

[13] Chen, X., & Li, S. (2021). Spatiotemporal trajectory stitching for multi-camera traffic surveillance. *Transportation Research Part C*.

[14] Quddus, M. A., Noland, R. B., & Bell, M. G. P. (2007). A multi-segment map-matching algorithm for vehicle navigation. *Transportation Research Part C*.

[15] Li, L., Li, Q., & Wang, S. (2013). Urban road trajectory matching with GPS data. *Journal of Intelligent Transportation Systems*.

[16] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.

[17] Jocher, G., Chirki, A., & Qiu, J. (2023). Ultralytics YOLOv8. *https://github.com/ultralytics/ultralytics*.

[18] Wang, G., Zhang, Y., Cheng, J., Liu, S., & Yang, Y. (2018). Group-Sort: Multi-object tracking by sorted group association. *AVSS*.

[19] Zheng, W., Huang, L., Deng, Y., & Liu, X. (2015). Person re-identification: Current methods and future directions. *IEEE Signal Processing Magazine*.

[20] Ye, M., Shen, J., Lin, G., Xiang, T., Shao, L., & Hoi, S. C. H. (2021). Deep learning for person re-identification: A survey and outlook. *IEEE TPAMI*.

[21] Luo, H., Gu, W., Liao, X., Lai, S., & Jiang, W. (2019). Bag of tricks and a strong baseline for deep person re-identification. *CVPR Workshops*.

[22] Ting, F., Zhu, J., & Xiang, T. (2022). Cross-modal retrieval with noisy correspondence. *ACM MM*.

[23] Gao, S., Zhang, C., & Li, J. (2023). Traffic camera network calibration for vehicle re-identification. *IEEE TITS*.

[24] Wang, Z., Tang, L., Liu, X., Yao, Z., Yi, S., Bai, J., ... & Wang, X. (2020). Spatial-temporal graph convolutional networks for trajectory prediction. *ICML*.

[25] Yu, F., Huang, Y., Wang, H., Luo, Y., Cui, X., & Yan, J. (2022). MOT20: The 2nd Multiple Object Tracking Challenge. *ECCV Workshops*.

[26] Luiten, J., Osep, A., Leal-Taixe, L., & Fischer, T. (2021). HOTA: A higher order metric for evaluating multi-object tracking. *IJCV*.

[27] Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS*.

[28] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.

[29] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.

[30] Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
