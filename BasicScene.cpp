#include "BasicScene.h"
#include <Eigen/src/Core/Matrix.h>
#include <edges.h>
#include <memory>
#include <per_face_normals.h>
#include <read_triangle_mesh.h>
#include <utility>
#include <vector>
#include "GLFW/glfw3.h"
#include "Mesh.h"
#include "PickVisitor.h"
#include "Renderer.h"
#include "ObjLoader.h"
#include "IglMeshLoader.h"

#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/edge_flaps.h"
#include "igl/loop.h"
#include "igl/upsample.h"
#include "igl/AABB.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/circulation.h"
#include "igl/edge_midpoints.h"
#include "igl/collapse_edge.h"
#include "igl/edge_collapse_is_valid.h"
#include "igl/write_triangle_mesh.h"

// initialization number of arms
int num_of_cyls = 7;
// minimal distance for reaching the destination
double delta = 0.05;
// high value make the reaching slower but smoother, low valuw makes it faster
double step_distance = 200;

bool ccdAnimate = false;
bool fabrikAnimate = false;
int IkSolver = 1; //start as CCD

using namespace cg3d;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create("camera", fov, float(width) / height, near, far);

    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{ std::make_shared<Material>("daylight", "shaders/cubemapShader") };
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{ Model::Create("background", Mesh::Cube(), daylight) };
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();


    auto program = std::make_shared<Program>("shaders/phongShader");
    auto program1 = std::make_shared<Program>("shaders/pickingShader");

    auto material{ std::make_shared<Material>("material", program) }; // empty material
    auto material1{ std::make_shared<Material>("material", program1) }; // empty material
    //    SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());

    material->AddTexture(0, "textures/box0.bmp", 2);
    auto sphereMesh{ IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj") };
    auto cylMesh{ IglLoader::MeshFromFiles("cyl_igl","data/xcylinder.obj") };
    auto cubeMesh{ IglLoader::MeshFromFiles("cube_igl","data/cube_old.obj") };
    sphere1 = Model::Create("sphere", sphereMesh, material);
    cube = Model::Create("cube", cubeMesh, material);

    //Axis
    Eigen::MatrixXd vertices(6, 3);
    vertices << -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1;
    Eigen::MatrixXi faces(3, 2);
    faces << 0, 1, 2, 3, 4, 5;
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Ones(6, 3);
    Eigen::MatrixXd textureCoords = Eigen::MatrixXd::Ones(6, 2);
    std::shared_ptr<Mesh> coordsys = std::make_shared<Mesh>("coordsys", vertices, faces, vertexNormals, textureCoords);
    axis.push_back(Model::Create("axis", coordsys, material1));
    axis[0]->mode = 1;
    axis[0]->Scale(3.2f, Axis::XYZ);
    root->AddChild(axis[0]);
    float scaleFactor = 1;
    cyls.push_back(Model::Create("cyl", cylMesh, material));
    cyls[0]->Scale(scaleFactor, Axis::X);
    cyls[0]->SetCenter(Eigen::Vector3f(-0.8f * scaleFactor, 0, 0));
    root->AddChild(cyls[0]);

    for (int i = 1; i < num_of_cyls; i++)
    {
        // create the cylinders and add to the vector cyls
        cyls.push_back(Model::Create("cyl", cylMesh, material));
        cyls[i]->Scale(scaleFactor, Axis::X);
        cyls[i]->Translate(1.6f * scaleFactor, Axis::X);
        cyls[i]->SetCenter(Eigen::Vector3f(-0.8f * scaleFactor, 0, 0));
        // important for the movement of all cyls together
        cyls[i - 1]->AddChild(cyls[i]);

        // add axis for each cylinder
        axis.push_back(Model::Create("axis", coordsys, material1));
        axis[i]->mode = 1;
        // define the size to twice link length
        axis[i]->Scale(3.2f, Axis::XYZ);
        cyls[i - 1]->AddChild(axis[i]);
    }
    // add axis for the last cylinder
    axis.push_back(Model::Create("axis", coordsys, material1));
    axis[num_of_cyls]->mode = 1;
    axis[num_of_cyls]->Scale(3.2f, Axis::XYZ);
    cyls[num_of_cyls - 1]->AddChild(axis[num_of_cyls]);

    cyls[0]->Translate({ 0.8f * scaleFactor,0,0 });
    //rotate first link by 90 degrees on z-axis
    cyls[0]->RotateByDegree(90, Eigen::Vector3f(0, 0, 1));

    auto morphFunc = [](Model* model, cg3d::Visitor* visitor) {
        return model->meshIndex;//(model->GetMeshList())[0]->data.size()-1;
    };
    autoCube = AutoMorphingModel::Create(*cube, morphFunc);


    sphere1->showWireframe = true;
    autoCube->Translate({ -6,0,0 });
    autoCube->Scale(1.5f);
    // render the sphere in (5,0,0)
    sphere1->Translate({ 5,0,0 });

    autoCube->showWireframe = true;
    camera->Translate(22, Axis::Z);
    root->AddChild(sphere1);
    //    root->AddChild(cyl);
    root->AddChild(autoCube);
    // points = Eigen::MatrixXd::Ones(1,3);
    // edges = Eigen::MatrixXd::Ones(1,3);
    // colors = Eigen::MatrixXd::Ones(1,3);

    // cyl->AddOverlay({points,edges,colors},true);
    cube->mode = 1;
    auto mesh = cube->GetMeshList();

    //autoCube->AddOverlay(points,edges,colors);
    // mesh[0]->data.push_back({V,F,V,E});
    int num_collapsed;

    // Function to reset original mesh and data structures
    V = mesh[0]->data[0].vertices;
    F = mesh[0]->data[0].faces;
    // igl::read_triangle_mesh("data/cube.off",V,F);
    igl::edge_flaps(F, E, EMAP, EF, EI);
    std::cout << "vertices: \n" << V << std::endl;
    std::cout << "faces: \n" << F << std::endl;

    std::cout << "edges: \n" << E.transpose() << std::endl;
    std::cout << "edges to faces: \n" << EF.transpose() << std::endl;
    std::cout << "faces to edges: \n " << EMAP.transpose() << std::endl;
    std::cout << "edges indices: \n" << EI.transpose() << std::endl;

}

void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 0.8f, 0.3f, 0.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 0.3f, 0.6f, 1.0f);
    program.SetUniform4f("Kdi", 0.5f, 0.5f, 0.0f, 1.0f);
    program.SetUniform1f("specular_exponent", 5.0f);
    program.SetUniform4f("light_position", 0.0, 15.0f, 0.0, 1.0f);
    //    cyl->Rotate(0.001f, Axis::Y);
    //cube->Rotate(0.1f, Axis::XYZ);

    // call to the algoritm
    if (IkSolver == 1)
    {
        CCDIK();
    }
    else if (IkSolver == 2)
    {
        FABRIK();
    }

}

void BasicScene::MouseCallback(Viewport* viewport, int x, int y, int button, int action, int mods, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event

    if (action == GLFW_PRESS) { // default mouse button press behavior
        PickVisitor visitor;
        visitor.Init();
        renderer->RenderViewportAtPos(x, y, &visitor); // pick using fixed colors hack
        auto modelAndDepth = visitor.PickAtPos(x, renderer->GetWindowHeight() - y);
        renderer->RenderViewportAtPos(x, y); // draw again to avoid flickering
        pickedModel = modelAndDepth.first ? std::dynamic_pointer_cast<Model>(modelAndDepth.first->shared_from_this()) : nullptr;
        pickedModelDepth = modelAndDepth.second;
        camera->GetRotation().transpose();
        xAtPress = x;
        yAtPress = y;

        // if (pickedModel)
        //     debug("found ", pickedModel->isPickable ? "pickable" : "non-pickable", " model at pos ", x, ", ", y, ": ",
        //           pickedModel->name, ", depth: ", pickedModelDepth);
        // else
        //     debug("found nothing at pos ", x, ", ", y);

        if (pickedModel && !pickedModel->isPickable)
            pickedModel = nullptr; // for non-pickable models we need only pickedModelDepth for mouse movement calculations later

        if (pickedModel)
            pickedToutAtPress = pickedModel->GetTout();
        else
            cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::ScrollCallback(Viewport* viewport, int x, int y, int xoffset, int yoffset, bool dragging, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event
    auto system = camera->GetRotation().transpose();
    if (pickedModel && pickedModel->name == "cyl")
    {
        // we need all the cyls to move together
        cyls[0]->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        pickedToutAtPress = pickedModel->GetTout();
    }
    else if (pickedModel)
    {
        pickedModel->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        pickedToutAtPress = pickedModel->GetTout();
    }
    else
    {
        // translate the  whole scene
        camera->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::CursorPosCallback(Viewport* viewport, int x, int y, bool dragging, int* buttonState)
{
    if (dragging) {
        auto system = camera->GetRotation().transpose() * GetRotation();
        auto moveCoeff = camera->CalcMoveCoeff(pickedModelDepth, viewport->width);
        auto angleCoeff = camera->CalcAngleCoeff(viewport->width);
        if (pickedModel && pickedModel->name == "cyl") {
            //pickedModel->SetTout(pickedToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                cyls[0]->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::X);
                pickedModel->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::Y);
            }
        }
        else if (pickedModel) {
            // camera->SetTout(cameraToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                pickedModel->TranslateInSystem(system * pickedModel->GetRotation(), { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Y);
                pickedModel->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::X);
            }
        }
        else {
            // camera->SetTout(cameraToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                root->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff / 10.0f, float(yAtPress - y) / moveCoeff / 10.0f, 0 });
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                root->RotateInSystem(system, float(x - xAtPress) / 180.0f, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                root->RotateInSystem(system, float(x - xAtPress) / angleCoeff, Axis::Y);
                root->RotateInSystem(system, float(y - yAtPress) / angleCoeff, Axis::X);
            }
        }
        xAtPress = x;
        yAtPress = y;
    }
}

void BasicScene::KeyCallback(Viewport* viewport, int x, int y, int key, int scancode, int action, int mods)
{
    auto system = camera->GetRotation().transpose();

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) // NOLINT(hicpp-multiway-paths-covered)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_UP:
            // rotates picked link around the current X axis. When nothing is picked rotate the whole scene.
            if (pickedModel && pickedModel != root)
            {
                pickedModel->RotateInSystem(system, 0.15f, Axis::X);
            }
            else
            {
                root->RotateInSystem(system, -0.15f, Axis::X);
            }
            //cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::X);
            break;
        case GLFW_KEY_DOWN:
            // rotates picked link around the current X axis. When nothing is picked rotate the whole scene.
            if (pickedModel && pickedModel != root)
            {
                pickedModel->RotateInSystem(system, -0.15f, Axis::X);
            }
            else
            {
                root->RotateInSystem(system, 0.15f, Axis::X);
            }
            //cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::X);
            break;
        case GLFW_KEY_LEFT:
            // rotates picked link around the previous link Y axis. When nothing is picked rotate the whole scene.
            if (pickedModel && pickedModel != root)
            {
                pickedModel->RotateInSystem(system, -0.15f, Axis::Y);
            }
            else
            {
                root->RotateInSystem(system, 0.15f, Axis::Y);
            }
            //cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::Y);
            break;
        case GLFW_KEY_RIGHT:
            // rotates picked link around the previous link Y axis. When nothing is picked rotate the whole scene.
            if (pickedModel && pickedModel != root)
            {
                pickedModel->RotateInSystem(system, 0.15f, Axis::Y);
            }
            else
            {
                root->RotateInSystem(system, -0.15f, Axis::Y);
            }
            //cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::Y);
            break;
        case GLFW_KEY_W:
            camera->TranslateInSystem(system, { 0, 0.1f, 0 });
            break;
        case GLFW_KEY_S:
            camera->TranslateInSystem(system, { 0, -0.1f, 0 });
            break;
        case GLFW_KEY_A:
            camera->TranslateInSystem(system, { -0.1f, 0, 0 });
            break;
        case GLFW_KEY_D:
            // prints destination position
            printDestination();
            //camera->TranslateInSystem(system, {0.1f, 0, 0});
            break;
        case GLFW_KEY_B:
            camera->TranslateInSystem(system, { 0, 0, 0.1f });
            break;
        case GLFW_KEY_F:
            camera->TranslateInSystem(system, { 0, 0, -0.1f });
            break;
        case GLFW_KEY_1:
            if (pickedIndex > 0)
                pickedIndex--;
            break;
        case GLFW_KEY_2:
            if (pickedIndex < cyls.size() - 1)
                pickedIndex++;
            break;
        case GLFW_KEY_3:
            if (tipIndex >= 0)
            {
                if (tipIndex == cyls.size())
                    tipIndex--;
                sphere1->Translate(GetSpherePos());
                tipIndex--;
            }
            break;
        case GLFW_KEY_4:
            if (tipIndex < cyls.size())
            {
                if (tipIndex < 0)
                    tipIndex++;
                sphere1->Translate(GetSpherePos());
                tipIndex++;
            }
            break;
        case GLFW_KEY_SPACE:
            // starts and stops IK solver animation
            ccdAnimate = !ccdAnimate;
            fabrikAnimate = !fabrikAnimate;
            break;
        case GLFW_KEY_C:
            // change between the kind of solvers
            if (IkSolver == 1)
            {
                IkSolver = 2;
                std::cout << "changed from ccd to fabrik\n";
            }
            else
            {
                IkSolver = 1;
                std::cout << "changed from fabrik to ccd\n";
            }
            break;
        case GLFW_KEY_P:
            // prints rotation matrices of the picked link, If nothing:prints the rotation matrix of the whole scene.
            printRotationMatrices();
            break;
        case GLFW_KEY_T:
            // prints arms tip positions
            printCylTips();
            break;
        case GLFW_KEY_N:
            // pick the next link, or the first one in case the last link is picked
            // if picked model is not cyl make error
            if (pickedModel)
            {
                if (pickedIndex < cyls.size() - 1)
                    pickedIndex++;
                else
                    pickedIndex = 0;
            }
            pickedModel = cyls[pickedIndex];
            break;
        }
    }
}

void BasicScene::printCylTips()
{
    // going through all the cyls
    for (int i = 0; i < num_of_cyls; i++)
    {
        Eigen::Matrix4f cyl_trans = cyls[i]->GetAggregatedTransform();
        Eigen::Vector3f cyl_center = Eigen::Vector3f(cyl_trans.col(3).x(), cyl_trans.col(3).y(), cyl_trans.col(3).z());
        Eigen::Vector3f cyl_tip_pos = cyl_center + cyls[i]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };
        printf("cylinder %d: (%.3f, %.3f, %.3f)\n", i, cyl_tip_pos.x(), cyl_tip_pos.y(), cyl_tip_pos.z());;
    }
}

void BasicScene::printRotationMatrices()
{
    Eigen::Matrix3f objRotation = root->GetRotation();
    for (int i = 0; i < num_of_cyls; i++)
    {
        if (cyls[i] == pickedModel)
        {
            objRotation = pickedModel->GetRotation();
            break;
        }
    }

    if (pickedModel != root)
    {
        Eigen::Vector3f euler = objRotation.eulerAngles(2, 0, 2) * (180.f / 3.14159f);
        std::cout << "Total Rotation: \n" << objRotation << std::endl;

        Eigen::Matrix3f phi;
        phi.row(0) = Eigen::Vector3f(cos(euler.x()), -sin(euler.x()), 0);
        phi.row(1) = Eigen::Vector3f(sin(euler.x()), cos(euler.x()), 0);
        phi.row(2) = Eigen::Vector3f(0, 0, 1);
        std::cout << "Phi Rotation: \n" << phi << std::endl;


        Eigen::Matrix3f theta;
        theta.row(0) = Eigen::Vector3f(1, 0, 0);
        theta.row(1) = Eigen::Vector3f(0, cos(euler.y()), -sin(euler.y()));
        theta.row(2) = Eigen::Vector3f(0, sin(euler.y()), cos(euler.y()));
        std::cout << "Theta Rotation: \n" << theta << std::endl;


        Eigen::Matrix3f psi;
        psi.row(0) = Eigen::Vector3f(cos(euler.z()), -sin(euler.z()), 0);
        psi.row(1) = Eigen::Vector3f(sin(euler.z()), cos(euler.z()), 0);
        psi.row(2) = Eigen::Vector3f(0, 0, 1);
        std::cout << "Psi Rotation: \n" << psi << std::endl;
    }
    else {
        std::cout << "objRotation: \n" << objRotation << std::endl;
    }
}

void BasicScene::printDestination()
{
    Eigen::Matrix4f dest = sphere1->GetAggregatedTransform();
    Eigen::Vector3f dest_pos = Eigen::Vector3f(dest.col(3).x(), dest.col(3).y(), dest.col(3).z());
    printf("Destination position: (%f, %f, %f)\n", dest_pos.x(), dest_pos.y(), dest_pos.z());
}

Eigen::Vector3f BasicScene::GetSpherePos()
{
    Eigen::Vector3f l = Eigen::Vector3f(1.6f, 0, 0);
    Eigen::Vector3f res;
    res = cyls[tipIndex]->GetRotation() * l;
    return res;
}

void BasicScene::CCDIK()
{
    if (!ccdAnimate)
        return;

    Eigen::Matrix4f dest = sphere1->GetAggregatedTransform();
    Eigen::Vector3f dest_pos = Eigen::Vector3f(dest.col(3).x(), dest.col(3).y(), dest.col(3).z());

    Eigen::Matrix4f first_trans = cyls[0]->GetAggregatedTransform();
    Eigen::Vector3f first_center = Eigen::Vector3f(first_trans.col(3).x(), first_trans.col(3).y(), first_trans.col(3).z());
    Eigen::Vector3f first_tip_pos = first_center - cyls[0]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

    Eigen::Matrix4f last_trans = cyls[num_of_cyls - 1]->GetAggregatedTransform();
    Eigen::Vector3f last_center = Eigen::Vector3f(last_trans.col(3).x(), last_trans.col(3).y(), last_trans.col(3).z());
    Eigen::Vector3f E_pos = last_center + cyls[num_of_cyls - 1]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

    // check if the distance is valid
    if (num_of_cyls * 1.6f < (dest_pos - first_tip_pos).norm())
    {
        printf("Destination is too far, please locate it closer\n");
        ccdAnimate = false;
        fabrikAnimate = false;
        return;
    }
    //check if reached destination
    double distance_to_dest = (dest_pos - E_pos).norm();
    if (distance_to_dest < delta)
    {
        printf("Done.\n distance: %f\n", (dest_pos - E_pos).norm());
        ccdAnimate = false;
        fabrikAnimate = false;
        return;
    }

    for (int i = num_of_cyls - 1; i >= 0; i--)
    {
        Eigen::Matrix4f cyl_trans = cyls[i]->GetAggregatedTransform();
        Eigen::Vector3f cyl_center = Eigen::Vector3f(cyl_trans.col(3).x(), cyl_trans.col(3).y(), cyl_trans.col(3).z());
        Eigen::Vector3f R_pos = cyl_center - cyls[i]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

        Eigen::Vector3f ER = (E_pos - R_pos).normalized();
        Eigen::Vector3f DR = (dest_pos - R_pos).normalized();

        Eigen::Vector3f normal = ER.cross(DR);

        double dot_product = DR.dot(ER);

        if (abs(dot_product) > 1) {
            dot_product = 1;
        }
        double angle_step = step_distance;
        double distance_to_dest = (dest_pos - E_pos).norm();

        // when getting close make the reaching faster
        if (distance_to_dest < 2) {
            angle_step = 100;
        }
        if (distance_to_dest < 0.75) {
            angle_step = 50;
        }
        if (distance_to_dest < 0.4) {
            angle_step = 20;
        }

        float total_angle = acosf(dot_product)*0.25 / angle_step;
        auto rotate_vector = (cyls[i]->GetRotation().transpose() * normal).normalized();
        auto rotate_matrix = (Eigen::AngleAxisf(total_angle, rotate_vector)).toRotationMatrix();
        // The sequence of three rotations that we saw in class Z-X-Z (3-1-3)
        auto euler = rotate_matrix.eulerAngles(2, 0, 2); 
        cyls[i]->Rotate(euler[0], Axis::Z);
        cyls[i]->Rotate(euler[1], Axis::X);
        cyls[i]->Rotate(euler[2], Axis::Z);
    }
}

void BasicScene::FABRIK() {
    if (!fabrikAnimate)
        return;

    //t - target position
    Eigen::Matrix4f dest = sphere1->GetAggregatedTransform();
    Eigen::Vector3f dest_pos = Eigen::Vector3f(dest.col(3).x(), dest.col(3).y(), dest.col(3).z());

    Eigen::Matrix4f first_trans = cyls[0]->GetAggregatedTransform();
    Eigen::Vector3f first_center = Eigen::Vector3f(first_trans.col(3).x(), first_trans.col(3).y(), first_trans.col(3).z());
    Eigen::Vector3f first_tip_pos = first_center - cyls[0]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

    Eigen::Matrix4f last_trans = cyls[num_of_cyls - 1]->GetAggregatedTransform();
    Eigen::Vector3f last_center = Eigen::Vector3f(last_trans.col(3).x(), last_trans.col(3).y(), last_trans.col(3).z());
    Eigen::Vector3f E_pos = last_center + cyls[num_of_cyls - 1]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };


    // check if the distance is valid
    if (num_of_cyls * 1.6f < (dest_pos - first_tip_pos).norm())
    {
        printf("Destination is too far, please locate it closer\n");
        ccdAnimate = false;
        fabrikAnimate = false;
        return;
    }
    // The joint positions p
    std::vector<Eigen::Vector3f> p;

    // Set disjoint positions (p_0 the is first disjoin)
    for (int i = 0; i < num_of_cyls; i++) {
        Eigen::Matrix4f trans = cyls[i]->GetAggregatedTransform();
        Eigen::Vector3f center = Eigen::Vector3f(trans.col(3).x(), trans.col(3).y(), trans.col(3).z());
        p.push_back(center - cyls[i]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 });
    }
    p.push_back(E_pos);

    // 1.15: set b as the initial pos of p
    auto b = p[0];

    // 1.17: calculate diff(distance to target)
    float diff = (E_pos - dest_pos).norm();

    if (diff < delta) {
        std::cout << "distance: " << diff << std::endl;
        ccdAnimate = false;
        fabrikAnimate = false;
        return;
    }
    while (diff > delta) {
        // 1.19: STAGE 1: FORWARD REACHING
        p[num_of_cyls] = dest_pos;

        //1.22:
        for (int i = num_of_cyls - 1; i >= 1; i--) {
            double ri = (p[i + 1] - p[i]).norm();
            double lambda = 1.6f / ri;
            p[i] = (1 - lambda)* p[i + 1] + lambda * p[i];
        }
        // STAGE 2: BACKWARD REACHING
        //1.32:
        for (int i = 0; i <= num_of_cyls-1; i++) {
            double ri = (p[i + 1] - p[i]).norm();
            double lambda = 1.6f / ri;
            p[i+1] = (1 - lambda) * p[i] + lambda * p[i+1];
        }
        //1.39: caculate diff and loop again
        diff = (p[num_of_cyls] - dest_pos).norm();
    }

    //rotate the cyls by a little bit using the previosly calculted loaction in p
    for (int i = 0; i <= num_of_cyls - 1; i++) {
        Eigen::Matrix4f cyl_trans = cyls[i]->GetAggregatedTransform();
        Eigen::Vector3f cyl_center = Eigen::Vector3f(cyl_trans.col(3).x(), cyl_trans.col(3).y(), cyl_trans.col(3).z());
        Eigen::Vector3f R_pos = cyl_center - cyls[i]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

        Eigen::Matrix4f E_trans = cyls[i]->GetAggregatedTransform();
        Eigen::Vector3f E_center = Eigen::Vector3f(E_trans.col(3).x(), E_trans.col(3).y(), E_trans.col(3).z());
        Eigen::Vector3f E_pos = E_center + cyls[i]->GetRotation() * Eigen::Vector3f{ 0.8f, 0, 0 };

        Eigen::Vector3f ER = (E_pos - R_pos).normalized();
        Eigen::Vector3f DR = (p[i+1] - R_pos).normalized();

        Eigen::Vector3f normal = ER.cross(DR);

        double dot_product = DR.dot(ER);

        if (abs(dot_product) > 1) {
            dot_product = 1;
        }

        double distance_to_dest = (p[i + 1] - E_pos).norm();
        double angle_step = step_distance;

        // when getting close make the reaching faster
        if (distance_to_dest < 2) {
            angle_step = 100;
        }
        if (distance_to_dest < 0.75) {
            angle_step = 50;
        }

        float total_angle = acosf(dot_product) * 0.5 / angle_step;
        auto rotate_vector = (cyls[i]->GetRotation().transpose() * normal).normalized();
        auto rotate_matrix = (Eigen::AngleAxisf(total_angle, rotate_vector)).toRotationMatrix();
        // The sequence of three rotations that we saw in class Z-X-Z (3-1-3)
        auto euler = rotate_matrix.eulerAngles(2, 0, 2);
        cyls[i]->Rotate(euler[0], Axis::Z);
        cyls[i]->Rotate(euler[1], Axis::X);
        cyls[i]->Rotate(euler[2], Axis::Z);
    }
}