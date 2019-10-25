using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine;

[RequireComponent(typeof(MeshRenderer))]
[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshCollider))]
public class TerrainChunk : MonoBehaviour
{
    Vector3Int chunkPosition;
    TerrainGenerator generator;
    Voxel[,,] voxels;
    Vector3Int gridSize;

    bool dirty;
    bool updating;
    
    // Mesh
    Mesh mesh;
    MeshFilter meshFilter;
    MeshRenderer meshRenderer;
    MeshCollider meshCollider;

    // For Avoid GC
    List<Vector3> vertices = new List<Vector3>();
    List<int> triangles = new List<int>();

    public bool Dirty => dirty;
    public bool Updating => updating;
    public Vector3Int ChunkPosition => chunkPosition;

    [BurstCompile]
    struct VoxelNoiseJob : IJobParallelFor
    {
        [ReadOnly] public Vector3 chunkWorldPosition;
        [ReadOnly] public Vector3Int cellSize;
        [ReadOnly] public Vector3Int gridSize;
        [ReadOnly] public float frequency;
        
        [WriteOnly] public NativeArray<Voxel> voxels;
        
        public void Execute(int index)
        {
            Vector3Int gridPosition = To3DIndex(index);
            Vector3 worldPosition = gridPosition + chunkWorldPosition;

            Vector3 World2DPosition = new Vector3(worldPosition.x, worldPosition.z);
            
            float density = -worldPosition.y;
            density += Noise.Perlin2DFractal(World2DPosition, frequency, 5) * 10f;
            density += Noise.Perlin2DFractal(World2DPosition, frequency * 0.5f, 3) * 50f;
            
            voxels[index] = new Voxel {Density = density};
        }
        
        Vector3Int To3DIndex(int index)
        {
            return new Vector3Int 
            {			
                z = index % gridSize.z,
                y = (index / gridSize.z) % gridSize.y,
                x = index / (gridSize.y * gridSize.z)
            };
        }
    }

    void Awake()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>();
        meshCollider = GetComponent<MeshCollider>();
        mesh = new Mesh();
    }

    void Start()
    {
        meshFilter.mesh = mesh;
    }

    public void Init(Vector3Int position, TerrainGenerator terrainGenerator)
    {
        chunkPosition = position;
        generator = terrainGenerator;

        if (!generator)
            return;

        gridSize = terrainGenerator.CellSize + Vector3Int.one;
        voxels = new Voxel[gridSize.x, gridSize.y, gridSize.z];
        meshRenderer.material = generator.TerrainMaterial;
        
        GenerateVoxelDensity();
        dirty = true;
    }
    
    void GenerateVoxelDensity()
    {
        if (!generator)
            return;

        Vector3 chunkWorldPosition = generator.ChunkToWorldNotScaled(chunkPosition);
        if (generator.EnableJob)
        {
            NativeArray<Voxel> nativeVoxels = new NativeArray<Voxel>(gridSize.x * gridSize.y * gridSize.z, Allocator.TempJob);

            VoxelNoiseJob noiseJob = new VoxelNoiseJob
            {
                chunkWorldPosition = chunkWorldPosition,
                cellSize = generator.CellSize,
                gridSize = gridSize,
                voxels = nativeVoxels,
                frequency = generator.Frequency
            };
            JobHandle noiseJobHandle = noiseJob.Schedule(gridSize.x * gridSize.y * gridSize.z, 32);
            noiseJobHandle.Complete();
            
            unsafe
            {
                fixed (void* voxelPointer = voxels)
                {
                    UnsafeUtility.MemCpy(voxelPointer, NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(nativeVoxels), voxels.Length * (long) UnsafeUtility.SizeOf<Voxel>());
                }
            }

            nativeVoxels.Dispose();
        }
        else
        {
            for (int x = 0; x < gridSize.x; x++)
            {
                for (int y = 0; y < gridSize.y; y++)
                {
                    for (int z = 0; z < gridSize.z; z++)
                    {
                        Vector3 worldPosition = new Vector3(x, y, z) + chunkWorldPosition;
                        float density = -worldPosition.y;
                        density += Noise.Perlin2DFractal(new Vector3(worldPosition.x, worldPosition.z), generator.Frequency, 5) * 25f;
                        voxels[x, y, z].Density = density;
                    }
                }
            }   
        }
    }

    public void UpdateMesh()
    {
        if (!generator)
            return;

        if (generator.EnableJob)
        {
            MarchingCubes.GenerateMarchingCubesWithJob(voxels, generator.CellSize, generator.ChunkScale, generator.EnableTriangleIndexing, vertices, triangles);
        }
        else
        {
            MarchingCubes.GenerateMarchingCubes(voxels, generator.CellSize, generator.ChunkScale, generator.EnableTriangleIndexing, vertices, triangles);
        }
        mesh.Clear();
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.RecalculateNormals();

        meshCollider.sharedMesh = mesh;
        dirty = false;
    }

    public void SetVoxel(Vector3Int worldGridPosition, float value)
    {
        Vector3Int gridPosition = new Vector3Int
        {
            x = worldGridPosition.x - chunkPosition.x * generator.CellSize.x,
            y = worldGridPosition.y - chunkPosition.y * generator.CellSize.y,
            z = worldGridPosition.z - chunkPosition.z * generator.CellSize.z
        };

        if (gridPosition.x < 0 || gridPosition.y < 0 || gridPosition.z < 0)
            return;
        
        voxels[gridPosition.x, gridPosition.y, gridPosition.z].Density = value;
        dirty = true;
    }

    public void AddVoxel(Vector3Int worldGridPosition, float value)
    {
        Vector3Int gridPosition = new Vector3Int
        {
            x = worldGridPosition.x - chunkPosition.x * generator.CellSize.x,
            y = worldGridPosition.y - chunkPosition.y * generator.CellSize.y,
            z = worldGridPosition.z - chunkPosition.z * generator.CellSize.z
        };

        if (gridPosition.x < 0 || gridPosition.y < 0 || gridPosition.z < 0)
            return;
        
        voxels[gridPosition.x, gridPosition.y, gridPosition.z].Density += value;
        dirty = true;
    }

    public Voxel GetVoxel(Vector3Int gridPosition)
    {
        return voxels[gridPosition.x + 1, gridPosition.y + 1,  gridPosition.z + 1];
    }

    void OnDrawGizmos()
    {
        if (generator && generator.EnableDebug)
        {
            Gizmos.color = Color.white;
            Gizmos.DrawWireCube(generator.ChunkToWorldCenter(chunkPosition), Vector3.Scale(generator.CellSize, generator.ChunkScale));
        }
    }
}