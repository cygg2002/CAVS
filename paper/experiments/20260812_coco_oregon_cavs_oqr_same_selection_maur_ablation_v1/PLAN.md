# Plan

- [ ] Freeze and hash the unified point results and CAVS-OQR selections per dataset.
- [ ] Build/load `current@640` MAUR predictions from the matching raw ROI and Edge cache.
- [ ] Re-evaluate the unchanged selections with Reversible and MAUR.
- [ ] Assert identical actions, bytes and capacity utilization for each paired row.
- [ ] Run 1,000 paired bootstrap replicates: COCO image; Oregon event and image.
- [ ] Write machine-readable artifacts and a Chinese interpretation report.
- [ ] Launch with four low-priority CPU workers without touching the source run.
