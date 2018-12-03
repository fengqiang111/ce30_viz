#ifndef CLOUD_CLUSTER_SCENE_H
#define CLOUD_CLUSTER_SCENE_H
#include <pcl/PointIndices.h>
#include "cloud_scene.h"
#include "export.h"

namespace ce30_pcviz
{
class API CloudClusterScene : public CloudScene
{
public:
    CloudClusterScene(
        std::shared_ptr<pcl::visualization::PCLVisualizer> visualizer);
    virtual ~CloudClusterScene();
    virtual void Update();
    void DrawClusterFrame(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                          std::vector<pcl::PointIndices>& cluster_indices);
    void DBSCAN_kdtree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                       std::vector<pcl::PointIndices>& cluster_indices,
                       float eps, int min_samples_size);
    void DBSCAN_octree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                       std::vector<pcl::PointIndices>& cluster_indices,
                       float eps, int min_samples_size);

protected:
    void AddCubicFrame(const float x_min, const float x_max,
                       const float y_min, const float y_max,
                       const float z_min, const float z_max);
    void ClearAllCubicFrames();
    void Erase() override;
private:
    std::string EdgeID();
    std::vector<std::string> cubic_frame_edge_ids_;
};
} // namespace ce30_pcviz

#endif // CLOUD_CLUSTER_SCENE_H
