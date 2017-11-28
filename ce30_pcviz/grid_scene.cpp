#include "grid_scene.h"
#include "grid_geometry.h"

using namespace std;
using namespace pcl::visualization;

namespace ce30_pcviz {
GridScene::GridScene(shared_ptr<PCLVisualizer> viz) : StaticScene(viz)
{
}

void GridScene::Show() {
  int width = 30;
  int height =30;
  float size = 1.0f;
  float x = height * size;
  float y = width * size / 2;
  GridGeometry grid_geo(width, height, x, y, size);
  auto horizontals = grid_geo.Horizontals();
  auto verticals = grid_geo.Verticals();
  int index = 0;
  for (auto& hor : horizontals) {
    Viz().addLine(hor.first, hor.second, "hl" + to_string(index++));
  }
  index = 0;
  for (auto& ver : verticals) {
    Viz().addLine(ver.first, ver.second, "vl" + to_string(index++));
  }

  const float offset = 0.5;
  auto right_border = grid_geo.RightBorder();
  for (int i = 0; i <= height; i += 10) {
    auto p = right_border[i];
    auto pp = p;
    pp.y -= offset;
    stringstream ss;
    ss << fixed << setprecision(1) << p.x;
    Viz().addText3D(
        ss.str() + "m", pp, 0.5, 1.0, 1.0, 1.0, "left_landmark" + to_string(i));
  }
}
} // namespace ce30_pcviz
