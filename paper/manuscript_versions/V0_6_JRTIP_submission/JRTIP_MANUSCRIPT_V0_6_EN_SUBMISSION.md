# CAFRU: Cost-Aware Frame- and ROI-Level Selective Offloading for Real-Time Edge-Cloud Object Detection on Low-Power Cameras

Author names to be completed

Affiliations to be completed

Corresponding author: name and active e-mail address to be completed

## Abstract

Low-power smart cameras can sustain local response with compact object
detectors, but their accuracy is constrained by compute, memory, and power.
Sending every frame to a higher-capacity cloud detector improves available
compute but continuously consumes uplink bandwidth and makes results sensitive
to network and queueing delay. This paper presents the Cost-Aware Frame-ROI
Uploader (CAFRU), a real-time edge-cloud detection system comprising an
edge-side YOLO26s detector, selective offloading, a bounded asynchronous queue,
and a YOLO26x cloud reviewer. CAFRU extracts inexpensive frame- and
region-level statistics from edge detections. Four linear heads estimate the
probability and expected utility gain of full-frame and region-of-interest
(ROI) review, after which a utility-minus-payload-cost rule selects full-frame
upload, ROI upload, or local output. Under a common deployment confidence
threshold on COCO val2017 and at JPEG quality factor 80, CAFRU increases mean
average precision
over intersection-over-union (IoU) thresholds 0.50–0.95 (mAP50–95) from
0.5350 for
edge-only detection to 0.5765 while uploading 42.46% of images. Amortized
compute is 79.7 GFLOPs/image, 40.6% below the 134.1 GFLOPs/image of
cloud-only YOLO26x@512. With a modeled 10-Mbps uplink and 50-ms round-trip
time, mean end-to-end latency is 83.6 ms, compared with 128.3 ms and
0.5996 mAP50–95 for cloud-only processing. A component-level T41 camera
prototype implements JPEG/HTTP transport and bounded asynchronous scheduling;
its approximate-feature linear evaluator requires 0.676 ms per decision. The
results show that
edge-cloud detection requires joint control of review value, payload
granularity, and result freshness rather than isolated model accuracy or
latency.

**Keywords:** edge-cloud collaboration; object detection; selective offloading; real-time video analytics; smart camera; resource-aware inference

## 1 Introduction

Low-power smart cameras are increasingly used for wildlife monitoring,
security inspection, and unattended sensing. Local inference reduces raw-video
transmission and preserves basic responsiveness during network outages.
However, memory, power, operator support, and compute limits restrict the
detectors that can run continuously on camera-class devices. Real-time
detectors provide increasingly favorable accuracy-latency trade-offs [1-5],
but substantial capacity gaps remain within a model family. Under the common
deployment-output evaluation used in this work, YOLO26s achieves 0.5350 COCO
mAP50–95, whereas YOLO26x@512 with JPEG quality 80 obtains 0.5996. Reviews of
edge inference likewise
identify quantization, memory access, compilation, and hardware operator
coverage as practical constraints beyond nominal FLOPs [6, 7].

A natural architecture therefore retains a lightweight edge detector and
deploys a higher-capacity reviewer in the cloud. Full offloading, however,
makes every frame pay image-encoding, uplink, round-trip, and cloud-service
costs. Network jitter and queue backlog can also make a successfully returned
detection obsolete for the frame that triggered it. Collaborative inference
has considered model partitioning, early exit, and task offloading [8-10],
while video-analytics systems have studied execution placement, frame
filtering, bandwidth adaptation, and server feedback [11-18]. Selective review
for object detection nevertheless differs from low-confidence offloading for
classification: each frame contains a variable set of localized objects, the
system may need either global or local review, and asynchronous returns must be
matched to a continuously advancing video stream.

Existing systems address complementary parts of this problem. Reducto filters
camera frames based on the relationship between content and task accuracy
[15]. RegionFilter uses server feedback to control region-aware encoding [19].
EdgeBoost performs confidence-based selective offloading for
resource-constrained classification [20]. These studies establish the value of
selective processing, but do not jointly address detector-level utility
supervision, full-frame versus ROI granularity, box-level fusion, and
deadline-aware return handling. Moreover, neural-network confidence is not
automatically a calibrated probability of correctness [21]. A fixed
low-confidence rule may upload detections that cloud review does not improve,
while missing edge-empty frames or confident errors. The uploader should
therefore learn whether cloud review improves object-level utility rather than
equating uncertainty with error.

This paper proposes the Cost-Aware Frame-ROI Uploader (CAFRU) and integrates it
with a bounded asynchronous transport path. The contributions are:

1. A detector-utility supervision method for dual-granularity offloading. Low
   cost frame and ROI features are labeled by the object-level utility
   difference before and after cloud review rather than by confidence alone.

2. A lightweight four-head uploader that predicts positive-gain probability
   and expected gain for full-frame and ROI review. Its outputs are combined
   with explicit payload cost to expose configurable accuracy, bandwidth, and
   amortized-compute operating points.

3. A real-time closed loop combining selective admission, bounded queues,
   keep-latest/drop-new scheduling, result-age checks, IoU-based box
   replacement, and class-aware NMS. This extends evaluation from isolated
   model latency to the freshness of results in continuous video.

4. An evaluation separating algorithmic evidence on full COCO from measured
   system costs on a T41 camera prototype, including ordinary ROI rules,
   payload regularization, cloud resolution, JPEG quality, queue scheduling,
   and runtime limitations.

## 2 Related work

### 2.1 Real-time detection and constrained deployment

YOLO formulated object detection as a unified single-stage regression problem
[1], and subsequent work has improved multi-scale representation, training,
and end-to-end inference [2-5]. YOLOv10 and RT-DETR pursue NMS-free YOLO and
end-to-end Transformer detection, respectively [3, 4]. We use YOLO26s and
YOLO26x as heterogeneous detectors from one model family. YOLO26 is currently
documented in a 2026 preprint [5]; its architecture is an experimental
component rather than a contribution of this paper.

Deployability depends on more than parameter count or FLOPs. Quantization
error, memory traffic, compiler support, and accelerator operators can dominate
edge performance [6]. Edge-intelligence research consequently advocates joint
placement across devices, edge servers, and clouds [7]. CAFRU is complementary
to model compression: either detector may be replaced, provided that the edge
model produces boxes and the reviewer can process full images or crops.

### 2.2 Collaborative inference and selective offloading

Neurosurgeon selects a deep neural network (DNN) partition according to device
and network cost [8].
BranchyNet exits simple samples at intermediate classifiers [9], and the
split-computing survey by Matsubara et al. summarizes the resulting
communication, energy, and accuracy trade-offs [10]. CAFRU does not transmit
intermediate tensors. It completes a usable detection locally, then sends
selected image pixels for review. This avoids requiring identical split
networks at both endpoints and permits independent cloud-model upgrades.

DeepDecision jointly chooses execution location and DNN configuration for
mobile video analytics [11], while LAVEA considers latency-aware edge video
services [12]. EdgeBoost demonstrates confidence-based selective offloading
for classification [20]. A classification request has one sample-level label,
whereas detection adds multi-object localization, class consistency, crop
remapping, and partial fusion. CAFRU trains from TP/FP utility change and
explicitly distinguishes full-frame and ROI actions.

### 2.3 Selective video processing and freshness

Edge video analytics must jointly control compute, bandwidth, and latency [13].
AWStream profiles accuracy-bandwidth relationships for wide-area network
(WAN) adaptation [14].
Reducto filters redundant camera frames [15], and NoScope uses specialized
cascades and difference detectors to accelerate video queries [16]. Glimpse
shows that server feedback arriving after a frame interval can degrade
continuous recognition and uses trigger frames and local caching to cope with
stale hints [17]. Edge-assisted mobile augmented reality similarly couples
cloud object detection with local processing and latency control [18].

RegionFilter exploits spatial redundancy by selecting different encoding
configurations for ROI and non-ROI regions [19]. Its regions are generated from
low-level features and historical server feedback. CAFRU instead starts from
current edge detections, adaptively expands candidate crops, and predicts their
object-detection utility. Reviews of near-edge detection and recent temporal
cloud-edge ensembles further show that placement and temporal coordination
remain active research topics [22, 23]. Our focus is a reproducible path from
COCO utility supervision to bounded scheduling on a camera-class prototype,
including negative evidence about the limited current contribution of ROI
review.

## 3 Method

### 3.1 Problem formulation and online architecture

For video frame \(I_t\), the edge detector \(D_e\) returns
\(B_e^t=\{b_i\}_{i=1}^{N_t}\), where
\(b_i=(x_1,y_1,x_2,y_2,c_i,p_i)\) contains image-coordinate bounds, class
label, and confidence. The cloud reviewer is denoted by \(D_c\). The uploader
selects an action from
\(\mathcal A_t=\{\mathrm{skip},\mathrm{frame},\mathrm{roi}\}\) to improve
detection quality subject to communication and timing costs.

The online system contains five stages: camera capture and YOLO26s inference;
CAFRU admission; JPEG encoding and a bounded queue; YOLO26x cloud review; and
result-age validation followed by detection fusion. A skipped frame
immediately emits \(B_e^t\). Full-frame or ROI requests are handled by a
separate upload worker so that capture and edge inference do not wait for the
cloud. Every response carries a frame identifier and timestamps. A response
enters the fusion cache only if its age is within the application deadline.

**[Fig. 1 placeholder: online edge-cloud architecture and asynchronous
feedback. Draw according to `FIGURE_DRAWING_SPEC_ZH.md`.]**

The decoded frame feeds both edge inference and pixel-level complexity
measurement, while edge boxes provide count, coverage, confidence, and ROI
statistics. An ROI is not merely an uploaded list of edge-box coordinates.
The corresponding image pixels are cropped around an adaptively expanded edge
box and transmitted with an offset. If no request is admitted, edge boxes are
reused directly. A late cloud return is discarded rather than retroactively
modifying a previously emitted frame.

### 3.2 Low-cost features and ROI construction

CAFRU introduces no additional convolutional feature network. The frame vector
\(x_f\in\mathbb R^{15}\) contains four groups: image edge density; detection
count and coverage; maximum, mean, minimum, and standard deviation of
confidence; and binary or aggregate risk statistics for empty detection,
sparsity, low coverage, mean/maximum box area, edge-touching boxes,
low-confidence boxes, and small boxes. Except for edge density, these features
are obtained in a linear scan of edge detections.

For each edge box, \(x_{r,i}\in\mathbb R^{15}\) contains confidence and its
complement, relative confidence, area and log area, aspect ratio, distance to
the image boundary, small/edge indicators, normalized center, coverage share,
and shared frame context. The image crop is

\[
R_i=\operatorname{clip}(\operatorname{expand}(b_i,\gamma_i),I_t),
\tag{1}
\]

where \(\gamma_i\) depends on box area and boundary distance. Small objects use
more context; boundary objects use clipped asymmetric crops. The payload is
\((\operatorname{JPEG}(R_i),t,\Delta x_i,\Delta y_i)\), and cloud boxes are
shifted by \((\Delta x_i,\Delta y_i)\) into original-image coordinates.

### 3.3 Utility supervision and the four prediction heads

During training, COCO annotation \(Y\) is used to measure utility before and
after review. Predictions and annotations are greedily matched one-to-one when
their classes agree and IoU is at least 0.5. Let \(TP\) and \(FP\) denote the
numbers of true positives and false positives, respectively. We define

\[
U(B,Y)=TP(B,Y)-\alpha FP(B,Y),
\tag{2}
\]

with \(\alpha=0.25\). The full-frame gain is

\[
g_f=\frac{U(B_c,Y)-U(B_e,Y)}{\max(1,|Y|)}.
\tag{3}
\]

For ROI candidate \(i\), remapped cloud boxes are fused with unaffected edge
boxes to obtain \(B_{fr}\), giving

\[
g_{r,i}=\frac{U(B_{fr},Y)-U(B_e,Y)}{\max(1,|Y|)}.
\tag{4}
\]

Normalization prevents crowded images from receiving larger targets solely
because they contain more objects. Binary positive-gain labels are
\(z=\mathbb 1[g>0]\). Thus, edge-empty frames, confident false positives, or
localization errors can become positive examples when cloud review actually
improves utility. Conversely, a low-confidence detection is not positive when
review provides no improvement.

The frozen uploader has four linear heads:

\[
p_f=\sigma(w_f^\top\tilde x_f+b_f),\qquad
\hat g_f=v_f^\top\tilde x_f+a_f,
\tag{5}
\]

\[
p_{r,i}=\sigma(w_r^\top\tilde x_{r,i}+b_r),\qquad
\hat g_{r,i}=v_r^\top\tilde x_{r,i}+a_r.
\tag{6}
\]

The probability heads use class-balanced logistic regression. The gain heads
use linear regression with Huber loss. Features are standardized using
training-split statistics only. A deterministic 90/10 train-validation split,
small feature noise, and L2 regularization are used for the policy heads.
The four heads contain only tens of scalar weights and require
standardization, dot products, and sigmoid evaluation after export.

**[Fig. 2 placeholder: offline utility-label generation and online CAFRU
decision. Draw according to `FIGURE_DRAWING_SPEC_ZH.md`.]**

Offline training and online inference are separate lanes in Fig. 2. The camera
loads frozen parameters and does not retrain on the live stream. A distribution
shift may motivate later calibration on independently labeled data, but that is
an offline model-maintenance operation.

### 3.4 Cost-aware decision and bounded scheduling

For full-frame and ROI candidates, CAFRU computes

\[
S_f=p_f\max(0,\hat g_f)-\lambda_b C_f,\qquad
S_{r,i}=p_{r,i}\max(0,\hat g_{r,i})-\lambda_b C_{r,i},
\tag{7}
\]

where \(C=k/100\), \(k\) is estimated JPEG size in KB, and \(\lambda_b\)
weights payload cost. A full frame is admitted when \(p_f\ge\tau_f\),
\(S_f>0\), and the queue can accept work. Otherwise, at most \(K\) top-scoring
ROIs satisfying \(p_{r,i}\ge\tau_r\) and \(S_{r,i}>0\) are admitted. The frame
is local-only if no candidate qualifies. Increasing \(\tau_f\) or \(\tau_r\)
reduces offloading but generally reduces review gain.

The trained heads currently consume image and detection statistics, not
network measurements. A separate runtime layer enforces queue capacity through
drop-new or keep-latest and discards returned results whose age exceeds the
deadline. Uplink rate and RTT affect observed queue occupancy and result age
but are not direct features of the trained heads. This separation distinguishes
estimated review utility from runtime result freshness and avoids retraining
the utility heads for every network condition.

For full-frame review, the current implementation uses cloud detections for
that frame; a skipped frame retains edge detections. ROI review replaces
IoU-matched boxes within the crop, may add sufficiently confident unmatched
cloud boxes, and then performs class-aware non-maximum suppression (NMS). The
gate can exploit complementary edge and cloud errors, but it does not
guarantee performance above cloud-only inference. We interpret only results
measured under a common evaluation path.

### 3.5 Complexity

Frame statistics, ROI features, and scoring are all \(O(N_t)\); each head has
fixed dimension 15. Policy parameters and arithmetic are negligible compared
with YOLO26s. Encoding, networking, and cloud inference occur only for admitted
requests. Amortized system compute is

\[
F_{\mathrm{sys}}=F_e+\frac{1}{M}\sum_{t=1}^{M}
\left(a_f^tF_c(s_f^t)+\sum_i a_{r,i}^tF_c(s_{r,i}^t)\right),
\tag{8}
\]

where \(F_e\) is edge-detector compute and \(F_c(s)\) is cloud-detector
compute at input size \(s\). Both full frames and crops are resized to the
configured cloud input size in the current implementation; cloud FLOPs are
therefore counted by request and input size, not scaled directly by crop area.

## 4 Experimental setup

### 4.1 Datasets, models, and splits

Standard accuracy experiments use COCO [24]. Utility labels are generated on
118,287 train2017 images using pretrained YOLO26s and YOLO26x as edge and
cloud detectors. The lightweight heads use a 90/10 split within train2017.
Final detection metrics are computed on the independent 5,000-image val2017
set, which is not used to generate utility labels or fit policy parameters.
The main evaluation covers all 80 COCO categories and re-evaluates edge,
cloud, and collaborative outputs at a common deployment confidence threshold.

The edge input size is 640, and both edge and cloud deployment confidence
thresholds are 0.25.
Cloud input sizes 512 and 640 are compared; 512 is the balanced default.
Deployment experiments simulate JPEG Q50-Q90 encode/decode. Exact commands,
threshold mappings, and result-source files are provided in Online Resource 1.
The recorded COCO cloud-inference timing runs use one NVIDIA RTX 4090. Because
latency is hardware-specific, GFLOPs/image is also reported as the
hardware-independent compute proxy.

The real prototype uses a T41 camera system on chip (SoC). YOLO26s is converted
through the INT8 neural-network accelerator (NNA) toolchain, the camera
provides NV12 frames, and selected frames are
JPEG-encoded and sent by HTTP multipart to a YOLO26x service. T41 experiments
measure capture, edge inference, encoding, HTTP, cloud inference, queueing, and
policy overhead. Their latency is not compared directly with workstation COCO
inference.

### 4.2 Baselines

- **Edge-only:** YOLO26s is used for every image without upload.

- **Cloud-only:** YOLO26x is used for every image.

- **Rule selective:** low confidence, small objects, boundary boxes, empty
  detection, and periodic keyframes trigger upload.

- **ROI-rule:** only rule-selected expanded ROIs are reviewed, testing whether
  ordinary ROI-plus-cloud review is sufficient.

- **CAFRU:** the proposed four-head utility model. Accuracy, Balanced,
  Efficient, and Bandwidth are operating points of one uploader, not different
  networks.

### 4.3 Metrics

Detection metrics are Precision, Recall, AP at IoU 0.50 (AP50), and mean AP
over IoU thresholds 0.50:0.05:0.95 (mAP50–95). Upload image ratio is the
fraction of input images generating at least one request. KB/image is total
JPEG payload divided by all input images, not only uploaded images; the logs
use 1 KB = 1024 bytes. GFLOPs/image denotes amortized theoretical edge plus
selective-cloud compute in billions of floating-point operations per image.

Modeled end-to-end (E2E) latency includes edge inference, encode/decode,
transfer, round-trip time, cloud inference, and queue waiting. Continuous-video
experiments also report P95 E2E, candidate drop ratio, and stale-result rate.
In the T41 scheduling-isolation runs, scheduling-loop frames per second (FPS)
is processed frames
divided by wall time. A timely-result frame has a cached cloud result whose age
does not exceed deadline \(D\), and effective FPS@D is timely-result frames
divided by wall time. One cloud return may serve multiple later frames while
it remains fresh, so timely-result frames may exceed completed requests.
These are system metrics rather than COCO detection metrics.

### 4.4 Evidence boundaries

An auxiliary set of 100 indoor T41 frames is checked against 108 automatically
generated SpeciesNet `person` boxes. These are SpeciesNet-assisted
pseudo-reference annotations, not manually verified ground truth. The check is
reported only as preliminary consistency evidence.

The T41 evaluator contains exported linear-head parameters but currently
approximates several Python features, including image complexity, detection
count normalization, and relative confidence. Exact Python/C++ feature
equivalence has not yet been established. Board-side policy latency and the
observed upload ratio are valid measurements of that prototype run, but are
not presented as an exact replication of the COCO Balanced point.

## 5 Results

### 5.1 Main COCO results

**Table 1 Main COCO val2017 results. All outputs are re-evaluated at the common
deployment confidence threshold of 0.25. JPEG codec and network latency are
evaluated separately in Table 3.**

| Method | Precision | Recall | mAP50 | mAP50-95 | Upload ratio | KB/image | GFLOPs/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Edge-only YOLO26s | 0.7147 | 0.5621 | 0.6672 | 0.5350 | 0.0000 | 0.00 | 22.8 |
| ROI-rule strict | 0.7300 | 0.5628 | 0.6681 | 0.5351 | 0.3824 | 1.50 | 74.1 |
| Rule selective | 0.7318 | 0.5832 | 0.6817 | 0.5486 | 0.6406 | 22.94 | 168.1 |
| CAFRU-Accuracy | 0.7835 | 0.6202 | 0.7251 | 0.5918 | 0.5656 | 34.90 | 98.7 |
| CAFRU-Balanced | 0.7758 | 0.6092 | 0.7163 | 0.5829 | 0.4248 | 26.55 | 79.8 |
| CAFRU-Efficient | 0.7671 | 0.5919 | 0.7032 | 0.5704 | 0.2442 | 15.87 | 55.6 |
| CAFRU-Bandwidth | 0.7519 | 0.5748 | 0.6861 | 0.5518 | 0.0920 | 6.28 | 35.2 |

Under the common deployment-output metric, CAFRU-Balanced improves edge-only
mAP50-95 by 0.0479 while uploading 42.48% of images. CAFRU-Accuracy yields the
largest gain, 0.0568, whereas CAFRU-Bandwidth limits uploads to 9.20% and still
improves mAP50-95 by 0.0168. The JPEG-encoded cloud-only upper reference and the
corresponding end-to-end latency are reported under the same output threshold
in Table 3.

**[Fig. 3: COCO deployment-output mAP50-95 versus KB/image, with bubble area
proportional to amortized GFLOPs/image.]**

This common-threshold comparison avoids treating detector outputs produced at
different confidence thresholds as directly comparable. Consequently, the
primary claim is not that CAFRU exceeds cloud-only accuracy, but that it recovers
part of the cloud accuracy gain with lower upload, compute, and latency costs.

### 5.2 Core ablation

**Table 2 Core CAFRU ablations. Rows in each group change only the stated
factor.**

| Group | Setting | mAP50-95 | Upload ratio | KB/image | GFLOPs/image | Interpretation |
|---|---|---:|---:|---:|---:|---|
| Cost | No payload cost | 0.5856 | 0.5184 | 31.44 | 92.3 | Slightly higher accuracy, more upload |
| Cost | CAFRU-Balanced | 0.5830 | 0.4246 | 26.55 | 79.7 | Suppresses low-value requests |
| Granularity | ROI-rule strict | 0.5351 | 0.3824 | 1.50 | 74.1 | Low bandwidth, insufficient accuracy |
| Branch | ROI-only | 0.5406 | 0.0020 | 0.03 | 23.1 | Insufficient candidate coverage |
| Branch | Frame-only | 0.5829 | 0.4238 | 26.53 | 79.6 | Dominant current source of gain |
| Branch | Frame+ROI | 0.5830 | 0.4246 | 26.55 | 79.7 | ROI is a limited supplement |
| Cloud size | 512 | 0.5830 | 0.4246 | 26.55 | 79.7 | Balanced default |
| Cloud size | 640 | 0.5880 | 0.4246 | 26.55 | 111.8 | +0.0050 mAP, +40.3% compute |
| JPEG | Q70 | 0.5744 | 0.4270 | 20.44 | 80.1 | Lower bandwidth |
| JPEG | Q80 | 0.5765 | 0.4246 | 26.55 | 79.7 | Balanced codec setting |

Each ablation group uses its corresponding independent rerun as the within-group
reference; minute differences are not compared across groups. The main
Balanced snapshot and the cost/cloud-size snapshots admit 2,124 and 2,123
tasks, respectively. One threshold-boundary ROI changes the rounded
mAP50-95 from 0.5829 to 0.5830 without affecting the group-wise conclusions.

Adding payload cost reduces upload ratio from 0.5184 to 0.4246 for only a
0.0026 mAP50-95 decrease. The best ordinary ROI rule sends only
1.50 KB/image but reaches 0.5351 mAP50-95, substantially below
CAFRU-Balanced. Ordinary ROI review cannot recover edge misses without a crop
proposal, small crops lose context, and remapping introduces fusion risk.

The branch ablation also establishes a limitation: current performance is
primarily due to learned full-frame selection. Frame+ROI improves only about
0.0001 mAP50-95 over Frame-only at the balanced point. We therefore treat ROI
as a fine-grained interface for future proposal and temporal methods rather
than the dominant present contribution. Increasing cloud size from 512 to 640
adds 0.0050 mAP50-95 but increases amortized compute by approximately 40.3%,
supporting the 512 default.

### 5.3 Weak-network latency and queueing

**Table 3 Deployment simulation at 10-Mbps uplink and 50-ms round-trip time
(RTT). Accuracy values use the common confidence threshold of 0.25.**

| Method | mAP50-95 | Upload ratio | KB/image | GFLOPs/image | Mean E2E (ms) | P95 E2E (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Edge-only | 0.5350 | 0.0000 | 0.00 | 22.8 | 29.5 | 29.5 |
| Cloud-only@512, JPEG quality 80 | 0.5996 | 1.0000 | 62.52 | 134.1 | 128.3 | 128.3 |
| ROI-rule strict | 0.5351 | 0.3824 | 1.50 | 74.1 | 62.7 | 122.6 |
| CAFRU Q70 | 0.5744 | 0.4270 | 20.44 | 80.1 | 79.0 | 162.5 |
| CAFRU Q80 | 0.5765 | 0.4246 | 26.55 | 79.7 | 83.6 | 179.0 |
| CAFRU Q90 | 0.5799 | 0.4238 | 37.73 | 79.6 | 92.5 | 207.3 |

At JPEG quality factor 80 (Q80), CAFRU improves edge-only mAP50-95 by 0.0415,
while remaining 0.0231 below cloud-only@512. Relative to full cloud offloading,
it reduces the upload ratio from 100% to 42.46%, amortized compute from 134.1 to
79.7 GFLOPs/image (40.6%), and mean E2E from 128.3 to 83.6 ms. Q90 adds 0.0034
mAP50-95 over Q80 but increases payload per image by 42.1% and pushes P95 above
200 ms. Q70 is a lower-bandwidth operating point rather than a universally
preferable configuration.

At 15 FPS and a 200-ms deadline, queue-0 cloud-only drops 50.00% of candidate
requests, whereas CAFRU Q70 drops 31.99%. Adding one keep-latest slot reduces
CAFRU Q70 candidate drops to 8.34%, but introduces 0.0860 stale returns per
image and increases P95 from 173.2 to 240.7 ms. Queue capacity therefore trades
admission rate against freshness and should be selected jointly with the
deadline.

**[Fig. 4: Mean and P95 E2E under 10-Mbps uplink and 50-ms RTT.]**

### 5.4 Selectivity, difficult subsets, and metric fairness

On the images selected by CAFRU Q80, edge, cloud, and fused mAP50-95 are
0.4756, 0.5536, and 0.5447, respectively. On the tiny-object subset, the
corresponding values are 0.4413, 0.5050, and 0.4834. The uploader therefore
concentrates requests on images that are difficult for both models but still
contain potential review utility. The fused result is not guaranteed to match
the cloud-only output because it combines edge retention, ROI replacement, and
class-aware suppression. On the non-uploaded subset, edge and fused mAP50-95
are 0.6192 and 0.6220, respectively; their 0.0028 difference provides a
negative control because images with no cloud request should not exhibit
material cloud gain.

These difficulty-oriented subsets are constructed from COCO annotation
geometry and upload events; they are not official COCO splits. Full
definitions, sample counts, and fairness tables are reported in Online
Resource 1. The fairness check also shows that official-confidence and
deployment-confidence evaluation change absolute mAP. Main claims compare
methods only under the same metric implementation.

### 5.5 T41 prototype

**Table 4 Measured T41 component costs. Rows are component measurements from
their corresponding runs and must not be summed as one-frame latency.**

| Component | Setting | Mean | P95/additional quantity | Interpretation |
|---|---|---:|---:|---|
| YOLO26s NNA | successful attempts | 585.971 ms | 701.825 ms | Main edge compute bottleneck |
| JPEG encoding | Q70, width 640 | 82.069 ms | 40.053 KB/upload | Measured board codec |
| HTTP round trip | Q70, width 640 | 1503.373 ms | - | Network and server wait included |
| Cloud YOLO26x | same run | 1123.246 ms | - | Measured service inference |
| Rule-selective loop | 100 frames | 37% upload | 8.462 KB/frame | 37/37 successful requests |
| Board linear evaluator | 50-frame online run | 0.676 ms | 0.817 ms | Approximate-feature prototype |

The rule-selective closed loop uploads 37 of 100 samples, all successfully.
Mean payload is 22.870 KB per upload and 8.462 KB amortized over all frames.
This demonstrates direct bandwidth reduction on the real JPEG/HTTP path.
Successful YOLO26s attempts average 585.971 ms, far above policy arithmetic;
the NNA runtime and cloud service, rather than the linear heads, dominate.

**Table 5 T41 scheduling-isolation experiment. These transport-only runs
exercise capture, JPEG/HTTP transport, bounded queues, and deadline gating;
local YOLO26s loading and inference are bypassed.**

| Mode | Queue | Deadline | Scheduling-loop FPS | Successful upload | Dropped tasks | Timely-result frames | Effective FPS@D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synchronous every-frame | 1 | 1000 ms | 0.614 | 19/19 | 0 | 0 | 0.000 |
| Asynchronous | 1 | 1000 ms | 6.623 | 11/11 | 89 | 62 | 4.106 |
| Asynchronous | 2 | 500 ms | 6.828 | 12/12 | 88 | 31 | 2.117 |
| Asynchronous | 2 | 1000 ms | 6.613 | 12/12 | 88 | 62 | 4.100 |
| Asynchronous | 2 | 2000 ms | 6.663 | 12/12 | 88 | 65 | 4.331 |
| Asynchronous | 4 | 1000 ms | 6.751 | 15/15 | 85 | 63 | 4.253 |

Synchronous every-frame upload is blocked by approximately 1.5-s HTTP round
trips, reaching a 0.614-FPS scheduling loop and returning no timely result
within 1000 ms. Asynchronous queues raise the isolated scheduling loop to
6.6-6.8 FPS, but maintaining freshness requires dropping 85-89 candidate
tasks. These rates exclude local YOLO26s inference and must not be interpreted
as full detector throughput. One returned result may remain valid for multiple
frames, so timely-result frames may exceed completed uploads. Queue 4 with a
1000-ms deadline yields 4.253 effective FPS@D. Relaxing the deadline from 500
to 2000 ms accepts older cached results and therefore raises the timely-frame
count.

**[Fig. 5 placeholder: T41 main thread, upload worker, and deadline gate. Draw
according to `FIGURE_DRAWING_SPEC_ZH.md`.]**

The SpeciesNet-assisted pseudo-reference check contains 100 frames and 108
automatically generated person boxes. Under this non-manual reference,
Edge, Cloud, and the current rule-loop Final obtain AP50 values of 0.2973,
0.4882, and 0.4835. This is preliminary evidence that cloud review may recover
edge person misses. Because Final is derived from a near-full-upload rule run
and the reference boxes are not manually verified, these values do not prove
CAFRU real-scene AP and are excluded from the primary claims.

## 6 Discussion

### 6.1 Comparison with ordinary ROI review

An ROI-only scheme needs an edge proposal as its crop center and therefore
cannot recover a complete edge miss. Small crops discard context, while large
crops approach full-frame cost. Fixed low-confidence, small-object, or boundary
rules identify possible difficulty but do not estimate whether cloud review
changes TP or FP. CAFRU instead learns the before/after utility difference and
compares full-frame and ROI actions against payload cost. ROI-rule strict is
highly bandwidth-efficient but remains 0.0478 mAP50-95 below
CAFRU-Balanced. The Frame-only ablation also prevents overclaiming: the present
advantage comes primarily from learned full-frame selection, with ROI providing
a limited fine-grained supplement.

### 6.2 Selecting an operating point

CAFRU exposes a Pareto set rather than one universally optimal configuration.
The Accuracy, Balanced, Efficient, and Bandwidth settings provide candidate
operating points with progressively stronger resource constraints. Their names
do not prescribe universal deployment choices. Thresholds and \(\lambda_b\)
should be selected
on an independent calibration set rather than the final test set. The runtime
queue then controls freshness according to current service capacity. These
layers separately answer whether review is valuable and whether it can finish
in time, improving interpretability and operational control.

### 6.3 Limitations and threats to validity

First, CAFRU is trained on COCO train2017 and tested on val2017. Wildlife,
traffic, low-light, or other domains may change edge-error distributions and
require independent calibration or retraining. Second, ROI review adds only
about 0.0001 mAP50-95 at the balanced point; stronger missed-object proposals,
context-aware crops, or temporal region memory are required. Third, the broad
network and queue study is simulated, while measured T41 sequences are short
and cover limited scenes. Fourth, intermittent NNA `SIGILL` failures remain,
and edge latency is reported for successful attempts; the prototype is not a
production-stability demonstration. Fifth, the C++ evaluator requires a
Python/C++ feature-equivalence test before exact policy portability can be
claimed. Sixth, the 100-frame SpeciesNet-assisted boxes are pseudo-reference
annotations and require manual verification before supporting formal
real-scene AP conclusions.

## 7 Conclusion

This paper presented CAFRU, a cost-aware frame- and region-of-interest-level
selective uploader for
real-time edge-cloud object detection on low-power cameras. Four lightweight
linear heads estimate positive-gain probability and expected gain for
full-frame and ROI review, while an explicit payload penalty, bounded queue,
and deadline gate control system cost and freshness. On COCO, CAFRU provides
configurable operating points between edge-only and cloud-only inference.
Under JPEG Q80 and the common deployment-output threshold, CAFRU improves
mAP50-95 from 0.5350 to 0.5765 with 42.46% upload and 79.7 GFLOPs/image,
compared with 0.5996 and 134.1 GFLOPs/image for cloud-only@512. Weak-network
results show that selective offloading must be
evaluated using end-to-end delay and timely results rather than isolated model
latency. Component-level T41 measurements validate JPEG/HTTP transport,
isolated asynchronous scheduling, and low-cost linear-head execution, while
exposing NNA instability, limited ROI gain, and incomplete real-scene
annotation. Future work will prioritize exact
board-feature equivalence, manually verified camera data, longer closed-loop
video evaluation, and temporal memory with dynamic network-cost calibration.

## Statements and Declarations

### Funding

To be completed by the authors. If no funding applies: The authors declare that
no funds, grants, or other support were received during the preparation of
this manuscript.

### Competing interests

The authors have no relevant financial or non-financial interests to disclose.

### Author contributions

To be completed using CRediT roles, including at minimum Conceptualization,
Methodology, Software, Validation, Writing-original draft, and Supervision.

### Data availability

COCO is available from its official distribution. Derived result tables,
configurations, and de-identified logs are planned for release in a public
repository after acceptance. Raw T41 images containing identifiable persons
will not be released before consent and anonymization requirements are
confirmed; de-identified statistics and reproduction scripts can be shared.

### Code availability

Repository and persistent archival link to be completed. An anonymized review
link should be supplied at submission, or the statement should specify release
upon acceptance.

### Ethics approval and consent

The indoor T41 images are used only for object-detection system evaluation and
not for identity recognition. Before submission, the authors must confirm
participant consent, image anonymization, and applicable institutional
requirements.

## References

[1] Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You Only Look
Once: Unified, Real-Time Object Detection. In: CVPR, pp. 779-788 (2016).
https://doi.org/10.1109/CVPR.2016.91

[2] Redmon, J., Farhadi, A.: YOLO9000: Better, Faster, Stronger. In:
CVPR, pp. 6517-6525 (2017). https://doi.org/10.1109/CVPR.2017.690

[3] Wang, A., Chen, H., Liu, L., Chen, K., Lin, Z., Han, J., Ding, G.:
YOLOv10: Real-Time End-to-End Object Detection. Adv. Neural Inf. Process.
Syst. 37 (2024)

[4] Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y.,
Chen, J.: DETRs Beat YOLOs on Real-time Object Detection. In: CVPR,
pp. 16965-16974 (2024). https://doi.org/10.1109/CVPR52733.2024.01605

[5] Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F.C., Kalfaoglu, M.E.:
Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.
arXiv:2606.03748 (2026). https://doi.org/10.48550/arXiv.2606.03748

[6] Shuvo, M.M.H., Islam, S.K., Cheng, J., Morshed, B.I.: Efficient
Acceleration of Deep Learning Inference on Resource-Constrained Edge
Devices: A Review. Proc. IEEE 111(1), 42-91 (2023).
https://doi.org/10.1109/JPROC.2022.3226481

[7] Zhou, Z., Chen, X., Li, E., Zeng, L., Luo, K., Zhang, J.: Edge
Intelligence: Paving the Last Mile of Artificial Intelligence With Edge
Computing. Proc. IEEE 107(8), 1738-1762 (2019).
https://doi.org/10.1109/JPROC.2019.2918951

[8] Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J.,
Tang, L.: Neurosurgeon: Collaborative Intelligence Between the Cloud and
Mobile Edge. In: ASPLOS, pp. 615-629 (2017).
https://doi.org/10.1145/3037697.3037698

[9] Teerapittayanon, S., McDanel, B., Kung, H.T.: BranchyNet: Fast
Inference via Early Exiting from Deep Neural Networks. In: ICPR,
pp. 2464-2469 (2016). https://doi.org/10.1109/ICPR.2016.7900006

[10] Matsubara, Y., Levorato, M., Restuccia, F.: Split Computing and Early
Exiting for Deep Learning Applications: Survey and Research Challenges.
ACM Comput. Surv. 55(5), 1-30 (2023).
https://doi.org/10.1145/3527155

[11] Ran, X., Chen, H., Zhu, X., Liu, Z., Chen, J.: DeepDecision: A Mobile
Deep Learning Framework for Edge Video Analytics. In: IEEE INFOCOM,
pp. 1421-1429 (2018). https://doi.org/10.1109/INFOCOM.2018.8485905

[12] Yi, S., Hao, Z., Zhang, Q., Zhang, Q., Shi, W., Li, Q.: LAVEA:
Latency-Aware Video Analytics on Edge Computing Platform. In: ACM/IEEE
SEC, pp. 1-13 (2017). https://doi.org/10.1145/3132211.3134459

[13] Xu, R., Razavi, S., Zheng, R.: Edge Video Analytics: A Survey on
Applications, Systems and Enabling Techniques. IEEE Commun. Surv. Tutor.
25(4), 2951-2982 (2023). https://doi.org/10.1109/COMST.2023.3323091

[14] Zhang, B., Jin, X., Ratnasamy, S., Wawrzynek, J., Lee, E.A.:
AWStream: Adaptive Wide-Area Streaming Analytics. In: ACM SIGCOMM,
pp. 236-252 (2018). https://doi.org/10.1145/3230543.3230554

[15] Li, Y., Padmanabhan, A., Zhao, P., Wang, Y., Xu, G.H., Netravali, R.:
Reducto: On-Camera Filtering for Resource-Efficient Real-Time Video
Analytics. In: ACM SIGCOMM, pp. 359-376 (2020).
https://doi.org/10.1145/3387514.3405874

[16] Kang, D., Emmons, J., Abuzaid, F., Bailis, P., Zaharia, M.: NoScope:
Optimizing Neural Network Queries over Video at Scale. Proc. VLDB Endow.
10(11), 1586-1597 (2017). https://doi.org/10.14778/3137628.3137664

[17] Chen, T.Y.-H., Ravindranath, L., Deng, S., Bahl, P., Balakrishnan, H.:
Glimpse: Continuous, Real-Time Object Recognition on Mobile Devices. In:
ACM SenSys, pp. 155-168 (2015).
https://doi.org/10.1145/2809695.2809711

[18] Liu, L., Li, H., Gruteser, M.: Edge Assisted Real-time Object Detection
for Mobile Augmented Reality. In: ACM MobiCom, pp. 1-16 (2019).
https://doi.org/10.1145/3300061.3300116

[19] Chen, T., Bu, Y., Zeng, Y., Xie, L., Lu, S.: RegionFilter:
Region-aware Video Filtering Mechanism on Resource-Constrained Edge
Nodes. Comput. Netw. 251, 110624 (2024).
https://doi.org/10.1016/j.comnet.2024.110624

[20] Said, N., Landsiedel, O.: EdgeBoost: Confidence Boosting for Resource
Constrained Inference via Selective Offloading. Comput. Netw. 269,
111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437

[21] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of
Modern Neural Networks. In: ICML, PMLR 70, pp. 1321-1330 (2017).
https://proceedings.mlr.press/v70/guo17a.html

[22] Setyanto, A., Sasongko, T.B., Fikri, M.A., Kim, I.K.: Near-Edge
Computing Aware Object Detection: A Review. IEEE Access 12, 2989-3011
(2024). https://doi.org/10.1109/ACCESS.2023.3347548

[23] Sudha, S.K., Aji, S.: TA-EnVODT: A Cloud-Edge Integrated
Temporal-Aware Ensemble for Real-Time Video Object Detection and Tracking
Using Knowledge Projection. Knowl.-Based Syst. 326, 114067 (2025).
https://doi.org/10.1016/j.knosys.2025.114067

[24] Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P.,
Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common Objects
in Context. In: ECCV, pp. 740-755 (2014).
https://doi.org/10.1007/978-3-319-10602-1_48
