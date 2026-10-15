# V0.6 Citation Audit

## 1. Verification policy

Bibliographic metadata was checked against primary publisher pages,
conference proceedings, Crossref DOI records, or official preprint pages.
The main manuscript cites only works used to support a specific claim.
YOLO26 is explicitly marked as a 2026 arXiv preprint.

## 2. Claim-to-citation map

| Manuscript claim | Supporting works | Scope guard |
|---|---|---|
| Real-time detectors trade accuracy for compute and deployment latency | YOLO, YOLOv10, RT-DETR, YOLO26 | YOLO26 is not yet a peer-reviewed journal/conference paper |
| Edge inference is constrained by memory, operators, quantization, and hardware support | Shuvo et al.; Zhou et al. | General deployment evidence, not T41-specific benchmarking |
| Collaborative inference may use partitioning, early exit, or task offloading | Neurosurgeon; BranchyNet; Matsubara et al. | These do not directly solve detector-level frame/ROI selection |
| Edge video analytics needs joint accuracy, bandwidth, and latency control | DeepDecision; LAVEA; Xu et al.; AWStream | System-level background |
| Selective frame filtering can reduce camera/network load | Reducto; NoScope | Primarily video query/filtering systems |
| Stale cloud feedback can harm continuous recognition | Glimpse; Liu et al. | Motivates bounded queues and freshness checks |
| Region-aware transmission can exploit spatial redundancy | RegionFilter | RegionFilter uses server feedback and encoding control; CAFRU uses edge detections and utility prediction |
| Confidence alone is not necessarily a calibrated correctness probability | Guo et al. | Calibration paper is classification-focused; used only to motivate learned utility |
| Confidence-based selective offloading is viable | EdgeBoost | EdgeBoost evaluates classification; it is not detector-level proof |
| Temporal cloud-edge detection is an active direction | TA-EnVODT | Different temporal ensemble design; not a direct baseline implementation |

## 3. Verified bibliography metadata

1. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You Only Look Once:
   Unified, Real-Time Object Detection. CVPR, 779-788 (2016).
   https://doi.org/10.1109/CVPR.2016.91
2. Redmon, J., Farhadi, A.: YOLO9000: Better, Faster, Stronger. CVPR,
   6517-6525 (2017). https://doi.org/10.1109/CVPR.2017.690
3. Wang, A., et al.: YOLOv10: Real-Time End-to-End Object Detection.
   Advances in Neural Information Processing Systems 37 (2024).
4. Zhao, Y., et al.: DETRs Beat YOLOs on Real-time Object Detection. CVPR,
   16965-16974 (2024). https://doi.org/10.1109/CVPR52733.2024.01605
5. Jocher, G., et al.: Ultralytics YOLO26: Unified Real-Time End-to-End
   Vision Models. arXiv:2606.03748 (2026).
   https://doi.org/10.48550/arXiv.2606.03748
6. Shuvo, M.M.H., Islam, S.K., Cheng, J., Morshed, B.I.: Efficient
   Acceleration of Deep Learning Inference on Resource-Constrained Edge
   Devices: A Review. Proceedings of the IEEE 111(1), 42-91 (2023).
   https://doi.org/10.1109/JPROC.2022.3226481
7. Zhou, Z., et al.: Edge Intelligence: Paving the Last Mile of Artificial
   Intelligence With Edge Computing. Proceedings of the IEEE 107(8),
   1738-1762 (2019). https://doi.org/10.1109/JPROC.2019.2918951
8. Kang, Y., et al.: Neurosurgeon: Collaborative Intelligence Between the
   Cloud and Mobile Edge. ASPLOS, 615-629 (2017).
   https://doi.org/10.1145/3037697.3037698
9. Teerapittayanon, S., McDanel, B., Kung, H.T.: BranchyNet: Fast Inference
   via Early Exiting from Deep Neural Networks. ICPR, 2464-2469 (2016).
   https://doi.org/10.1109/ICPR.2016.7900006
10. Matsubara, Y., Levorato, M., Restuccia, F.: Split Computing and Early
    Exiting for Deep Learning Applications: Survey and Research Challenges.
    ACM Computing Surveys 55(5), 1-30 (2023).
    https://doi.org/10.1145/3527155
11. Ran, X., et al.: DeepDecision: A Mobile Deep Learning Framework for Edge
    Video Analytics. IEEE INFOCOM, 1421-1429 (2018).
    https://doi.org/10.1109/INFOCOM.2018.8485905
12. Yi, S., et al.: LAVEA: Latency-Aware Video Analytics on Edge Computing
    Platform. ACM/IEEE SEC, 1-13 (2017).
    https://doi.org/10.1145/3132211.3134459
13. Xu, R., Razavi, S., Zheng, R.: Edge Video Analytics: A Survey on
    Applications, Systems and Enabling Techniques. IEEE Communications
    Surveys & Tutorials 25(4), 2951-2982 (2023).
    https://doi.org/10.1109/COMST.2023.3323091
14. Zhang, B., et al.: AWStream: Adaptive Wide-Area Streaming Analytics. ACM
    SIGCOMM, 236-252 (2018). https://doi.org/10.1145/3230543.3230554
15. Li, Y., et al.: Reducto: On-Camera Filtering for Resource-Efficient
    Real-Time Video Analytics. ACM SIGCOMM, 359-376 (2020).
    https://doi.org/10.1145/3387514.3405874
16. Kang, D., et al.: NoScope: Optimizing Neural Network Queries over Video
    at Scale. Proceedings of the VLDB Endowment 10(11), 1586-1597 (2017).
    https://doi.org/10.14778/3137628.3137664
17. Chen, T.Y.-H., et al.: Glimpse: Continuous, Real-Time Object Recognition
    on Mobile Devices. ACM SenSys, 155-168 (2015).
    https://doi.org/10.1145/2809695.2809711
18. Liu, L., Li, H., Gruteser, M.: Edge Assisted Real-time Object Detection
    for Mobile Augmented Reality. ACM MobiCom, 1-16 (2019).
    https://doi.org/10.1145/3300061.3300116
19. Chen, T., et al.: RegionFilter: Region-aware Video Filtering Mechanism on
    Resource-Constrained Edge Nodes. Computer Networks 251, 110624 (2024).
    https://doi.org/10.1016/j.comnet.2024.110624
20. Said, N., Landsiedel, O.: EdgeBoost: Confidence Boosting for Resource
    Constrained Inference via Selective Offloading. Computer Networks 269,
    111437 (2025). https://doi.org/10.1016/j.comnet.2025.111437
21. Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On Calibration of Modern
    Neural Networks. ICML, PMLR 70, 1321-1330 (2017).
22. Setyanto, A., et al.: Near-Edge Computing Aware Object Detection: A
    Review. IEEE Access 12, 2989-3011 (2024).
    https://doi.org/10.1109/ACCESS.2023.3347548
23. Sudha, S.K., Aji, S.: TA-EnVODT: A Cloud-Edge Integrated Temporal-Aware
    Ensemble for Real-Time Video Object Detection and Tracking Using
    Knowledge Projection. Knowledge-Based Systems 326, 114067 (2025).
    https://doi.org/10.1016/j.knosys.2025.114067
24. Lin, T.-Y., et al.: Microsoft COCO: Common Objects in Context. ECCV,
    740-755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

## 4. Corrections relative to V0.5

1. The Neurosurgeon DOI is `10.1145/3037697.3037698`, not the malformed DOI
   used in V0.5.
2. YOLO9000 pages are 6517-6525 in the DOI metadata.
3. The Shuvo review is a 2023 Proceedings of the IEEE article, volume 111,
   issue 1, pages 42-91.
4. The Matsubara survey is volume 55, issue 5 (2023), although it was first
   available online in 2022.
5. The peer-reviewed journal version of the edge-video-analytics survey is
   used instead of an arXiv-only survey where possible.
6. Claims about confidence are narrowed: confidence is treated as an
   imperfect utility proxy, not as proof that every low-confidence detection
   is erroneous.
7. EdgeBoost is described as classification-oriented selective offloading;
   it is not presented as a detector-level baseline.

## 5. Items requiring author confirmation before submission

1. Final author names, affiliations, corresponding-author e-mail, and ORCID.
2. Whether the YOLO26 preprint is acceptable to cite at submission time or
   should be updated if a peer-reviewed version appears.
3. Public repository/DOI for code and derived result tables.
4. Ethics/consent statement for the indoor T41 person images.

