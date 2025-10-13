#include <cmath>
#include <cstdint>
#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <limits>
#include <string>
#include <omp.h>
#include <cstring>

/**
 * @file lbmSimulation.cc
 * @brief A 2D Lattice Boltzmann (LBM) simulation for computational fluid dynamics.
 *
 * This simulation uses a D2Q9 LBM model to simulate fluid flow, including
 * collision, streaming, and bounce-back boundary conditions.
 */

enum BarrierType
{
    barrierOffsetMid,   ///< Barrier offset at mid-height.
    barrierCenterGap    ///< Barrier with a central gap.
};

enum FluidProperty
{
    density,    ///< Use density as the fluid property.
    speed,      ///< Use speed as the fluid property.
    vorticity   ///< Use vorticity as the fluid property.
};

// -----------------------------------------------------------------------------
// Data structure to hold all relevant LBM fields
// -----------------------------------------------------------------------------
struct LBMData
{
    // Simulation grid dimensions.
    uint32_t dimX;         ///< Number of grid cells in the x-direction.
    uint32_t dimY;         ///< Number of grid cells in the y-direction.

    // Contiguous storage for distribution functions and derived data.
    // The layout of the 14 subarrays (each of size dimX * dimY) is as follows:
    //   f0, fN, fE, fS, fW, fNE, fNW, fSE, fSW,
    //   density, velocityX, velocityY, vorticity, speed
    std::unique_ptr<double[]> data;  ///< Big array of size 14 * (dimX * dimY)

    // Pointers into the 'data' array:
    double* f0;         ///< Rest distribution function.
    double* fN;         ///< North distribution function.
    double* fE;         ///< East distribution function.
    double* fS;         ///< South distribution function.
    double* fW;         ///< West distribution function.
    double* fNE;        ///< North-East distribution function.
    double* fNW;        ///< North-West distribution function.
    double* fSE;        ///< South-East distribution function.
    double* fSW;        ///< South-West distribution function.
    double* density;    ///< Macroscopic density.
    double* velocityX;  ///< Macroscopic velocity in the x-direction.
    double* velocityY;  ///< Macroscopic velocity in the y-direction.
    double* vorticity;  ///< Vorticity of the flow.
    double* speed;      ///< Speed (magnitude of velocity).

    // Barrier array (true if the cell is a barrier).
    std::unique_ptr<bool[]> barrier;
};

// -----------------------------------------------------------------------------
// createLBMData()
// -----------------------------------------------------------------------------
LBMData createLBMData(uint32_t width, uint32_t height)
{
    LBMData lbmData;

    lbmData.dimX = width;
    lbmData.dimY = height;

    // Total number of cells in the simulation grid.
    const uint32_t size = lbmData.dimX * lbmData.dimY;
    lbmData.data = std::unique_ptr<double[]>(new double[14 * size]);

    // Assign pointers to each region in the contiguous block.
    lbmData.f0         = lbmData.data.get() + 0 * size;
    lbmData.fN         = lbmData.data.get() + 1 * size;
    lbmData.fE         = lbmData.data.get() + 2 * size;
    lbmData.fS         = lbmData.data.get() + 3 * size;
    lbmData.fW         = lbmData.data.get() + 4 * size;
    lbmData.fNE        = lbmData.data.get() + 5 * size;
    lbmData.fNW        = lbmData.data.get() + 6 * size;
    lbmData.fSE        = lbmData.data.get() + 7 * size;
    lbmData.fSW        = lbmData.data.get() + 8 * size;
    lbmData.density    = lbmData.data.get() + 9  * size;
    lbmData.velocityX  = lbmData.data.get() + 10 * size;
    lbmData.velocityY  = lbmData.data.get() + 11 * size;
    lbmData.vorticity  = lbmData.data.get() + 12 * size;
    lbmData.speed      = lbmData.data.get() + 13 * size;

    // Allocate the barrier array.
    lbmData.barrier = std::unique_ptr<bool[]>(new bool[size]);

    return lbmData;
}

// -----------------------------------------------------------------------------
// destroyLBMData()
// -----------------------------------------------------------------------------
void destroyLBMData(LBMData& /*lbmData*/)
{
    // Memory is automatically freed by unique_ptr.
}

// -----------------------------------------------------------------------------
// setEquilibrium()
// -----------------------------------------------------------------------------
static void setEquilibrium(LBMData& lbmData,
                           int x, int y,
                           double newVelocityX,
                           double newVelocityY,
                           double newDensity)
{
    const int idx = y * lbmData.dimX + x;

    const double oneNinth       = 1.0 / 9.0;
    const double fourNinths     = 4.0 / 9.0;
    const double oneThirtySixth = 1.0 / 36.0;

    const double velocity3X = 3.0 * newVelocityX;
    const double velocity3Y = 3.0 * newVelocityY;
    const double velocityX2 = newVelocityX * newVelocityX;
    const double velocityY2 = newVelocityY * newVelocityY;
    const double velocity2XY = 2.0 * newVelocityX * newVelocityY;
    const double velocity2 = velocityX2 + velocityY2;
    const double velocity2Factor = 1.5 * velocity2;

    lbmData.f0[idx]  = fourNinths * newDensity * (1.0 - velocity2Factor);
    lbmData.fE[idx]  = oneNinth * newDensity * (1.0 + velocity3X + 4.5 * velocityX2 - velocity2Factor);
    lbmData.fW[idx]  = oneNinth * newDensity * (1.0 - velocity3X + 4.5 * velocityX2 - velocity2Factor);
    lbmData.fN[idx]  = oneNinth * newDensity * (1.0 + velocity3Y + 4.5 * velocityY2 - velocity2Factor);
    lbmData.fS[idx]  = oneNinth * newDensity * (1.0 - velocity3Y + 4.5 * velocityY2 - velocity2Factor);
    lbmData.fNE[idx] = oneThirtySixth * newDensity * (1.0 + velocity3X + velocity3Y
                                                   + 4.5 * (velocity2 + velocity2XY) - velocity2Factor);
    lbmData.fSE[idx] = oneThirtySixth * newDensity * (1.0 + velocity3X - velocity3Y
                                                   + 4.5 * (velocity2 - velocity2XY) - velocity2Factor);
    lbmData.fNW[idx] = oneThirtySixth * newDensity * (1.0 - velocity3X + velocity3Y
                                                   + 4.5 * (velocity2 - velocity2XY) - velocity2Factor);
    lbmData.fSW[idx] = oneThirtySixth * newDensity * (1.0 - velocity3X - velocity3Y
                                                   + 4.5 * (velocity2 + velocity2XY) - velocity2Factor);

    lbmData.density[idx]   = newDensity;
    lbmData.velocityX[idx] = newVelocityX;
    lbmData.velocityY[idx] = newVelocityY;
}

// -----------------------------------------------------------------------------
// initBarrier()
// -----------------------------------------------------------------------------
void initBarrier(LBMData& lbmData, BarrierType barrierType)
{
    for (int j = 0; j < static_cast<int>(lbmData.dimY); j++)
    {
        const int row = j * lbmData.dimX;
        for (int i = 0; i < static_cast<int>(lbmData.dimX); i++)
        {
            switch (barrierType)
            {
            case barrierOffsetMid:
                if ((i == static_cast<int>(lbmData.dimX) / 8 ||
                     i == static_cast<int>(lbmData.dimX) / 8 + 1) &&
                    j > 8 * static_cast<int>(lbmData.dimY) / 27 &&
                    j < 17 * static_cast<int>(lbmData.dimY) / 27)
                {
                    lbmData.barrier[row + i] = true;
                }
                else
                {
                    lbmData.barrier[row + i] = false;
                }
                break;

            case barrierCenterGap:
                if ((i == static_cast<int>(lbmData.dimX) / 8 ||
                     i == static_cast<int>(lbmData.dimX) / 8 + 1) &&
                    ((j > 8 * static_cast<int>(lbmData.dimY) / 27 &&
                      j < 12 * static_cast<int>(lbmData.dimY) / 27) ||
                     (j > 13 * static_cast<int>(lbmData.dimY) / 27 &&
                      j < 17 * static_cast<int>(lbmData.dimY) / 27)))
                {
                    lbmData.barrier[row + i] = true;
                }
                else
                {
                    lbmData.barrier[row + i] = false;
                }
                break;

            default:
                lbmData.barrier[row + i] = false;
                break;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// initFluid()
// -----------------------------------------------------------------------------
void initFluid(LBMData& lbmData, double speed)
{
    for (int j = 0; j < static_cast<int>(lbmData.dimY); j++)
    {
        const int row = j * lbmData.dimX;
        for (int i = 0; i < static_cast<int>(lbmData.dimX); i++)
        {
            setEquilibrium(lbmData, i, j, speed, 0.0, 1.0);
            lbmData.vorticity[row + i] = 0.0;
        }
    }
}

// -----------------------------------------------------------------------------
// collide()
// -----------------------------------------------------------------------------
void collide(LBMData& lbmData, double viscosity)
{
    const double omega = 1.0 / (3.0 * viscosity + 0.5); // Relaxation parameter

    #pragma omp parallel for schedule(static, 16)
    for (int j = 1; j < static_cast<int>(lbmData.dimY) - 1; j++)
    {
        const int row = j * lbmData.dimX;
        #pragma omp simd
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++)
        {
            const int idx = row + i;

            // Compute macroscopic density.
            lbmData.density[idx] =
                lbmData.f0[idx] + lbmData.fN[idx] + lbmData.fS[idx] +
                lbmData.fE[idx] + lbmData.fW[idx] +
                lbmData.fNW[idx] + lbmData.fNE[idx] + lbmData.fSW[idx] + lbmData.fSE[idx];

            // Compute macroscopic velocities.
            lbmData.velocityX[idx] =
                (lbmData.fE[idx] + lbmData.fNE[idx] + lbmData.fSE[idx] -
                 lbmData.fW[idx] - lbmData.fNW[idx] - lbmData.fSW[idx]) /
                lbmData.density[idx];

            lbmData.velocityY[idx] =
                (lbmData.fN[idx] + lbmData.fNE[idx] + lbmData.fNW[idx] -
                 lbmData.fS[idx] - lbmData.fSE[idx] - lbmData.fSW[idx]) /
                lbmData.density[idx];

            // Precompute equilibrium coefficients.
            const double oneNinthDensity       = (1.0 / 9.0)  * lbmData.density[idx];
            const double fourNinthsDensity     = (4.0 / 9.0)  * lbmData.density[idx];
            const double oneThirtySixthDensity = (1.0 / 36.0) * lbmData.density[idx];

            const double velocity3X = 3.0 * lbmData.velocityX[idx];
            const double velocity3Y = 3.0 * lbmData.velocityY[idx];
            const double velocityX2 = lbmData.velocityX[idx] * lbmData.velocityX[idx];
            const double velocityY2 = lbmData.velocityY[idx] * lbmData.velocityY[idx];
            const double velocity2XY = 2.0 * lbmData.velocityX[idx] * lbmData.velocityY[idx];
            const double velocity2 = velocityX2 + velocityY2;
            const double velocity2Factor = 1.5 * velocity2;

            // Relaxation toward equilibrium for each distribution function.
            lbmData.f0[idx] += omega * (fourNinthsDensity * (1.0 - velocity2Factor) - lbmData.f0[idx]);
            lbmData.fE[idx] += omega * (oneNinthDensity * (1.0 + velocity3X + 4.5 * velocityX2 - velocity2Factor) - lbmData.fE[idx]);
            lbmData.fW[idx] += omega * (oneNinthDensity * (1.0 - velocity3X + 4.5 * velocityX2 - velocity2Factor) - lbmData.fW[idx]);
            lbmData.fN[idx] += omega * (oneNinthDensity * (1.0 + velocity3Y + 4.5 * velocityY2 - velocity2Factor) - lbmData.fN[idx]);
            lbmData.fS[idx] += omega * (oneNinthDensity * (1.0 - velocity3Y + 4.5 * velocityY2 - velocity2Factor) - lbmData.fS[idx]);
            lbmData.fNE[idx] += omega * (oneThirtySixthDensity * (1.0 + velocity3X + velocity3Y
                            + 4.5 * (velocity2 + velocity2XY) - velocity2Factor) - lbmData.fNE[idx]);
            lbmData.fSE[idx] += omega * (oneThirtySixthDensity * (1.0 + velocity3X - velocity3Y
                            + 4.5 * (velocity2 - velocity2XY) - velocity2Factor) - lbmData.fSE[idx]);
            lbmData.fNW[idx] += omega * (oneThirtySixthDensity * (1.0 - velocity3X + velocity3Y
                            + 4.5 * (velocity2 - velocity2XY) - velocity2Factor) - lbmData.fNW[idx]);
            lbmData.fSW[idx] += omega * (oneThirtySixthDensity * (1.0 - velocity3X - velocity3Y
                            + 4.5 * (velocity2 + velocity2XY) - velocity2Factor) - lbmData.fSW[idx]);
        }
    }
}

// -----------------------------------------------------------------------------
// stream()
// -----------------------------------------------------------------------------
void stream(LBMData& lbmData)
{
    // Process NW + NE directions (top to bottom)
    for (int j = static_cast<int>(lbmData.dimY) - 2; j > 0; j--) {
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++) {
            const int row  = j * lbmData.dimX;
            const int rowp = (j - 1) * lbmData.dimX;

            // NW direction (left -> right)
            lbmData.fN[row + i]  = lbmData.fN[rowp + i];
            lbmData.fNW[row + i] = lbmData.fNW[rowp + i + 1];

            // NE direction (right -> left)
            const int reversed_i = (lbmData.dimX - 1) - i;
            lbmData.fE[row + reversed_i]  = lbmData.fE[row + reversed_i - 1];
            lbmData.fNE[row + reversed_i] = lbmData.fNE[rowp + reversed_i - 1];
        }
    }

    // Process SE + SW directions (bottom to top)
    for (int j = 1; j < static_cast<int>(lbmData.dimY) - 1; j++) {
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++) {
            const int row  = j * lbmData.dimX;
            const int rown = (j + 1) * lbmData.dimX;

            // SE direction (right -> left)
            const int reversed_i = (lbmData.dimX - 1) - i;
            lbmData.fS[row + reversed_i]  = lbmData.fS[rown + reversed_i];
            lbmData.fSE[row + reversed_i] = lbmData.fSE[rown + reversed_i - 1];

            // SW direction (left -> right)
            lbmData.fW[row + i]  = lbmData.fW[row + i + 1];
            lbmData.fSW[row + i] = lbmData.fSW[rown + i + 1];
        }
    }
}

// -----------------------------------------------------------------------------
// bounceBackStream()
// -----------------------------------------------------------------------------
void bounceBackStream(LBMData& lbmData)
{
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < static_cast<int>(lbmData.dimY) - 1; j++)
    {
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++)
        {
            const int row  = j * lbmData.dimX;
            const int rowp = (j - 1) * lbmData.dimX;  // previous row
            const int rown = (j + 1) * lbmData.dimX;  // next row
            const int idx = row + i;                  // index of cell

            // Temp variables - Create thread-local copies
            double temp_fN = lbmData.fN[idx];
            double temp_fE = lbmData.fE[idx];
            double temp_fS = lbmData.fS[idx];
            double temp_fW = lbmData.fW[idx];
            double temp_fNE = lbmData.fNE[idx];
            double temp_fNW = lbmData.fNW[idx];
            double temp_fSE = lbmData.fSE[idx];
            double temp_fSW = lbmData.fSW[idx];

            // Bounce-back from neighboring barriers.
            if (lbmData.barrier[row + i - 1])   temp_fE = lbmData.fW[row + i - 1];
            if (lbmData.barrier[row + i + 1])   temp_fW = lbmData.fE[row + i + 1];
            if (lbmData.barrier[rowp + i])      temp_fN = lbmData.fS[rowp + i];
            if (lbmData.barrier[rown + i])      temp_fS = lbmData.fN[rown + i];
            if (lbmData.barrier[rowp + i - 1])  temp_fNE = lbmData.fSW[rowp + i - 1];
            if (lbmData.barrier[rowp + i + 1])  temp_fNW = lbmData.fSE[rowp + i + 1];
            if (lbmData.barrier[rown + i - 1])  temp_fSE = lbmData.fNW[rown + i - 1];
            if (lbmData.barrier[rown + i + 1])  temp_fSW = lbmData.fNE[rown + i + 1];

            // Safely swap value because all writes are to the current cell only
            lbmData.fE[idx] = temp_fE;
            lbmData.fW[idx] = temp_fW;
            lbmData.fN[idx] = temp_fN;
            lbmData.fS[idx] = temp_fS;
            lbmData.fNE[idx] = temp_fNE;
            lbmData.fNW[idx] = temp_fNW;
            lbmData.fSE[idx] = temp_fSE;
            lbmData.fSW[idx] = temp_fSW;
        }
    }
}

// -----------------------------------------------------------------------------
// checkStability()
// -----------------------------------------------------------------------------
bool checkStability(const LBMData& lbmData)
{
    bool stable = true;
    const int midY = lbmData.dimY / 2;
    const int row = midY * lbmData.dimX;

    for (int i = 0; i < static_cast<int>(lbmData.dimX); i++)
    {
        if (lbmData.density[row + i] <= 0)
        {
            stable = false;
            break;
        }
    }
    return stable;
}

// -----------------------------------------------------------------------------
// computeSpeed()
// -----------------------------------------------------------------------------
void computeSpeed(LBMData& lbmData)
{
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < static_cast<int>(lbmData.dimY) - 1; j++)
    {
        const int row = j * lbmData.dimX;
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++)
        {
            const int idx = row + i;
            lbmData.speed[idx] = std::sqrt(lbmData.velocityX[idx] * lbmData.velocityX[idx] +
                                           lbmData.velocityY[idx] * lbmData.velocityY[idx]);
        }
    }
}

// -----------------------------------------------------------------------------
// computeVorticity()
// -----------------------------------------------------------------------------
void computeVorticity(LBMData& lbmData)
{
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < static_cast<int>(lbmData.dimY) - 1; j++)
    {
        const int row  = j * lbmData.dimX;
        const int rowp = (j - 1) * lbmData.dimX;
        const int rown = (j + 1) * lbmData.dimX;

        #pragma omp simd
        for (int i = 1; i < static_cast<int>(lbmData.dimX) - 1; i++)
        {
            // Approximate finite differences (ignoring constant factors).
            lbmData.vorticity[row + i] =
                lbmData.velocityY[row + i + 1] - lbmData.velocityY[row + i - 1] -
                lbmData.velocityX[rown + i]    + lbmData.velocityX[rowp + i];
        }
    }
}

// -----------------------------------------------------------------------------
// Getter functions for demonstration
// -----------------------------------------------------------------------------
bool* getBarrier(LBMData& lbmData) { return lbmData.barrier.get(); }
double* getDensity(LBMData& lbmData) { return lbmData.density; }
double* getVorticity(LBMData& lbmData) { return lbmData.vorticity; }
double* getSpeed(LBMData& lbmData) { return lbmData.speed; }

// -----------------------------------------------------------------------------
// mapDensityToColor()
// -----------------------------------------------------------------------------
void mapDensityToColor(double value,
    uint8_t& r, uint8_t& g, uint8_t& b,
    bool isBarrier,
    double minValue,
    double maxValue)
{
    if (isBarrier)
    {
        // For barriers, use red.
        r = 255; g = 0; b = 0;
        return;
    }

    double deltaBelow = 1.0 - minValue;
    double deltaAbove = maxValue - 1.0;
    double delta = std::max(deltaBelow, deltaAbove);
    if (delta <= 0) delta = 1.0;

    double deviation = std::fabs(value - 1.0) / delta;
    deviation = std::min(1.0, deviation);
    uint8_t grey = static_cast<uint8_t>(255 * (1.0 - deviation));

    r = grey;
    g = grey;
    b = grey;
}

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
/**
 * @brief Entry point for the LBM simulation.
 *
 * If the command-line flag "--movie" is provided, a series of frames are saved with unique filenames.
 * Otherwise (default debug mode), a single image file is updated every timestep.
 */
int main(int argc, char** argv)
{
    int num_threads = 16;
    omp_set_num_threads(num_threads);

    // Determine if movie mode is enabled.
    bool movieMode = false;
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--movie")
        {
            movieMode = true;
            break;
        }
    }

    // Simulation grid dimensions and timing.
    uint32_t dimX         = 1920;  ///< Grid width.
    uint32_t dimY         = 1024;  ///< Grid height.
    uint32_t timeSteps    = 10000; // Total simulation timesteps.
    uint32_t saveInterval = 10;    ///< Save frame every saveInterval timesteps in movie mode.

    // Selected barrier and fluid property to monitor.
    BarrierType selectedBarrier         = barrierCenterGap;
    FluidProperty selectedFluidProperty = vorticity;

    // Physical properties.
    double physicalDensity   = 1380.0; // kg/m^3
    double physicalSpeed     = 1.0;    // m/s
    double physicalLength    = 2.0;    // m
    double physicalViscosity = 0.75;   // Pa·s
    double physicalTime      = 0.8;    // s
    double physicalFreq      = 0.04;   // s
    double reynoldsNumber    = (physicalDensity * physicalSpeed * physicalLength) / physicalViscosity;

    // Convert physical properties into simulation properties.
    double simulationDx        = physicalLength / static_cast<double>(dimY);
    double simulationDt        = physicalTime / static_cast<double>(timeSteps);
    double simulationSpeed     = (simulationDt / simulationDx) * physicalSpeed;
    double simulationViscosity = simulationDt / (simulationDx * simulationDx * reynoldsNumber);


    std::cout << std::fixed << std::setprecision(6)
              << "LBM-CFD> speed: "     << simulationSpeed
              << ", viscosity: "        << simulationViscosity
              << ", reynolds: "         << reynoldsNumber
              << std::endl << std::endl;

    // Create the LBM data structure.
    LBMData lbmData = createLBMData(dimX, dimY);
    initBarrier(lbmData, selectedBarrier);
    initFluid(lbmData, simulationSpeed);

    // Get pointers to simulation arrays.
    double* densityArr = getDensity(lbmData);
    bool* barrierArr   = getBarrier(lbmData);

    // Parameters for mapping density to color.
    double minValue = 0.370;
    double maxValue = 1.757;

    // Open CSV file for timing output.
    std::ofstream csvFile("fluid.csv");
    if (!csvFile.is_open())
    {
        std::cerr << "Error: Could not open fluid.csv for writing." << std::endl;
        return 1;
    }
    csvFile << "Timestep,SimStepTime,ImageConversionTime,ImageWriteTime\n";

    // Simulation loop variables.
    int outputCount = 0;
    double outputTime = 0.0;
    auto totalStart = std::chrono::high_resolution_clock::now();

    for (uint32_t t = 0; t < timeSteps; t++)
    {
        double conversionTime = 0.0;
        double writeTime = 0.0;

        // Determine whether to write a frame.
        // In movie mode, write only every saveInterval timesteps.
        // In debug mode, write a frame on every timestep (overwriting the same file).
        bool doFrame = movieMode ? (t % saveInterval == 0) : true;
        if (doFrame)
        {
            std::ostringstream oss;
            if (movieMode)
            {
                oss << "movieFrames/fluidframe" << std::setw(5) << std::setfill('0') << t << ".ppm";
            }
            else
            {
                oss << "fluidframe.ppm";
            }
            std::string filename = oss.str();

            std::vector<unsigned char> imageBuffer(dimX * dimY * 3);

            auto convStart = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for collapse(2)
            for (uint32_t j = 0; j < dimY; ++j)
            {
                for (uint32_t i = 0; i < dimX; ++i)
                {
                    uint32_t index = j * dimX + i;
                    double value   = densityArr[index];
                    bool isBar     = barrierArr[index];
                    uint8_t r, g, b;

                    mapDensityToColor(value, r, g, b, isBar, minValue, maxValue);
                    size_t pixelIndex = (j * dimX + i) * 3;
                    imageBuffer[pixelIndex    ] = r;
                    imageBuffer[pixelIndex + 1] = g;
                    imageBuffer[pixelIndex + 2] = b;
                }
            }
            auto convEnd = std::chrono::high_resolution_clock::now();
            conversionTime = std::chrono::duration<double>(convEnd - convStart).count();

            auto writeStart = std::chrono::high_resolution_clock::now();
            std::ofstream file(filename, std::ios::binary);
            if (file.is_open())
            {
                file << "P6\n" << dimX << " " << dimY << "\n255\n";
                file.write(reinterpret_cast<const char*>(imageBuffer.data()), imageBuffer.size());
                file.close();
                // Only print "Saved frame" message in movie mode.
                if (movieMode)
                    std::cout << "Saved frame to " << filename << "\n";
            }
            else
            {
                std::cerr << "Error: Could not open file " << filename << "\n";
            }
            auto writeEnd = std::chrono::high_resolution_clock::now();
            writeTime = std::chrono::duration<double>(writeEnd - writeStart).count();
        }

        // Print progress.
        if (static_cast<double>(t) * simulationDt >= outputTime)
        {
            if (!movieMode)
            {
                // Overwrite the same line in debug mode.
                std::cout << "\r" << std::fixed << std::setprecision(3)
                          << "LBM-CFD> time: " << static_cast<double>(t) * simulationDt
                          << " / " << physicalTime
                          << " , time step: " << t << " / " << timeSteps << std::flush;
            }
            else
            {
                std::cout << std::fixed << std::setprecision(3)
                          << "LBM-CFD> time: " << static_cast<double>(t) * simulationDt
                          << " / " << physicalTime
                          << " , time step: " << t << " / " << timeSteps << std::endl;
            }

            bool stable = checkStability(lbmData);
            if (!stable)
            {
                std::cerr << "\nLBM-CFD> Warning: simulation has become unstable (more time steps needed)\n";
            }

            switch (selectedFluidProperty)
            {
            case density:
                break;
            case speed:
                computeSpeed(lbmData);
                break;
            case vorticity:
                computeVorticity(lbmData);
                break;
            }
            outputCount++;
            outputTime = outputCount * physicalFreq;
        }

        auto simStart = std::chrono::high_resolution_clock::now();
        collide(lbmData, simulationViscosity);
        stream(lbmData);
        bounceBackStream(lbmData);
        auto simEnd = std::chrono::high_resolution_clock::now();
        double simStepTime = std::chrono::duration<double>(simEnd - simStart).count();

        csvFile << t << "," 
                << simStepTime << ","
                << conversionTime << ","
                << writeTime << "\n";
    }

    auto totalEnd = std::chrono::high_resolution_clock::now();
    double totalSimTime = std::chrono::duration<double>(totalEnd - totalStart).count();
    std::cout << "\nTotal simulation time: " << totalSimTime << " s\n";

    csvFile.close();


    csvFile.open("runtimes/" + std::to_string(num_threads) + "threads_runtime.csv");

    if (!csvFile.is_open())
    {
        std::cerr << "Error: Could not open runtime.csv for writing." << std::endl;
        return 1;
    }

    csvFile << "Threads,TotalSimTime\n";
    csvFile << num_threads << "," << totalSimTime << "\n";

    destroyLBMData(lbmData);
    return 0;
}
