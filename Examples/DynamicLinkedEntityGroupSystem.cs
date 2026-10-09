using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Transforms;

// Minimal adapter for the tag referenced by OptimizedTransforms.cs.
[BurstCompile]
[UpdateInGroup(typeof(SimulationSystemGroup))]
[UpdateAfter(typeof(TransformSystemGroup))]
public partial struct DynamicLinkedEntityGroupSystem : ISystem
{
    public partial struct HierarchyChanged : IComponentData
    {
    }

    EntityQuery _changedQuery;

    [BurstCompile]
    public void OnCreate(ref SystemState state)
    {
        _changedQuery = new EntityQueryBuilder(Allocator.Temp)
            .WithAll<HierarchyChanged>()
            .WithOptions(EntityQueryOptions.IncludeDisabledEntities)
            .Build(ref state);
        state.RequireForUpdate(_changedQuery);
    }

    [BurstCompile]
    public void OnUpdate(ref SystemState state)
    {
        var ecb = SystemAPI.GetSingleton<EndSimulationEntityCommandBufferSystem.Singleton>().CreateCommandBuffer(state.WorldUnmanaged);
        state.Dependency = new ClearHierarchyChangedJob { ecb = ecb.AsParallelWriter() }.ScheduleParallel(_changedQuery, state.Dependency);
    }

    [BurstCompile]
    partial struct ClearHierarchyChangedJob : IJobEntity
    {
        public EntityCommandBuffer.ParallelWriter ecb;

        public void Execute(Entity entity, [ChunkIndexInQuery] int sortKey, in HierarchyChanged changed)
        {
            ecb.RemoveComponent<HierarchyChanged>(sortKey, entity);
        }
    }
}
