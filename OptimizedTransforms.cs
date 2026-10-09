using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics.GraphicsIntegration;
using Unity.Transforms;

namespace OptimizedTransforms
{
    // ============================================================================
    // DEPTH LEVEL TAG COMPONENTS (0 bytes each - pure tags for archetypes)
    // ============================================================================

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth1 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth2 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth3 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth4 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth5 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth6 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth7 : IComponentData { }

    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepth8 : IComponentData { }

    /// <summary>
    /// Tag for entities deeper than depth 8. Updated via recursive traversal from depth-8 roots.
    /// </summary>
    [WriteGroup(typeof(LocalToWorld))]
    public struct TransformDepthDeep : IComponentData { }

    /// <summary>
    /// Marks entities that need depth tag recalculation (enableable = no structural change)
    /// </summary>
    [WriteGroup(typeof(LocalToWorld))]
    internal struct DepthDirty : IComponentData, IEnableableComponent { }

    // ============================================================================
    // DEPTH MAINTENANCE SYSTEM - Only runs on Parent changes
    // ============================================================================

    [UpdateInGroup(typeof(TransformSystemGroup))]
    [UpdateAfter(typeof(ParentSystem))]
    [UpdateBefore(typeof(UltraFastLocalToWorldSystem))]
    [BurstCompile]
    public partial struct DepthTagMaintenanceSystem : ISystem
    {
        private EntityQuery _newChildrenQuery;
        private EntityQuery _parentChangedQuery;
        private EntityQuery _depthDirtyQuery;
        private EntityQuery _lostParentQuery;

        // Whether DepthDirty may be enabled anywhere. Only this system enables it, so on frames when nothing was marked the
        // enabled-bit scan of every child chunk is skipped.
        private bool _dirtyPending;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static void TryDisableBuiltinLocalToWorld(ref SystemState state)
        {
            var world = state.WorldUnmanaged;
            world.GetExistingSystemState<Unity.Transforms.LocalToWorldSystem>().Enabled = false;
        }

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            var queryBuilder = new EntityQueryBuilder(Allocator.Temp)
                                    .WithAll<Parent, LocalTransform>()
                                    .WithNone<DepthDirty, TransformDepth1, TransformDepth2, TransformDepth3, TransformDepth4>()
                                    .WithNone<TransformDepth5, TransformDepth6, TransformDepth7, TransformDepth8, TransformDepthDeep>()
                                    .WithOptions(EntityQueryOptions.IncludeDisabledEntities);
            _newChildrenQuery = state.GetEntityQuery(queryBuilder);

            var changedQueryBuilder = new EntityQueryBuilder(Allocator.Temp)
                                            .WithAll<Parent>()
                                            .WithOptions(EntityQueryOptions.IncludeDisabledEntities);
            _parentChangedQuery = state.GetEntityQuery(changedQueryBuilder);
            _parentChangedQuery.SetChangedVersionFilter(ComponentType.ReadOnly<Parent>());

            var depthDirtyBuilder = new EntityQueryBuilder(Allocator.Temp)
                                        .WithAll<DepthDirty>()
                                        .WithOptions(EntityQueryOptions.IncludeDisabledEntities);
            _depthDirtyQuery = state.GetEntityQuery(depthDirtyBuilder);

            _lostParentQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                                    .WithAll<LocalTransform>()
                                    .WithNone<Parent>()
                                    .WithAny<TransformDepth1, TransformDepth2, TransformDepth3, TransformDepth4>()
                                    .WithAny<TransformDepth5, TransformDepth6, TransformDepth7, TransformDepth8>()
                                    .WithAny<TransformDepthDeep>()
                                    .WithOptions(EntityQueryOptions.IncludeDisabledEntities));

            state.RequireForUpdate<LocalTransform>();
            _dirtyPending = true;
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            TryDisableBuiltinLocalToWorld(ref state);

            var dependency = state.Dependency;
            var ecb = new EntityCommandBuffer(Allocator.TempJob);

            var depthDirtyLookup = SystemAPI.GetComponentLookup<DepthDirty>(true);
            var childLookup = SystemAPI.GetBufferLookup<Child>(true);

            if (!_newChildrenQuery.IsEmpty)
            {
                var newEntities = _newChildrenQuery.ToEntityArray(Allocator.Temp);
                foreach (var entity in newEntities)
                {
                    if (!depthDirtyLookup.HasComponent(entity))
                    {
                        ecb.AddComponent<DepthDirty>(entity);
                    }
                    ecb.SetComponentEnabled<DepthDirty>(entity, true);
                }
                newEntities.Dispose();
            }

            if (!_parentChangedQuery.IsEmpty)
            {
                var changedEntities = _parentChangedQuery.ToEntityArray(Allocator.Temp);
                foreach (var entity in changedEntities)
                {
                    if (!depthDirtyLookup.HasComponent(entity))
                        ecb.AddComponent<DepthDirty>(entity);

                    ecb.SetComponentEnabled<DepthDirty>(entity, true);
                }
                changedEntities.Dispose();
            }

            if (!_lostParentQuery.IsEmpty)
            {
                var lostParentEntities = _lostParentQuery.ToEntityArray(Allocator.Temp);
                foreach (var entity in lostParentEntities)
                {
                    RemoveAllDepthTags(entity, state.EntityManager, ecb);

                    if (childLookup.TryGetBuffer(entity, out var children))
                    {
                        MarkChildrenDirty(children, childLookup, ecb);
                    }
                }
                lostParentEntities.Dispose();
            }

            _dirtyPending |= !ecb.IsEmpty;
            ecb.Playback(state.EntityManager);
            ecb.Dispose();

            if (_dirtyPending)
            {
                _dirtyPending = !_depthDirtyQuery.IsEmpty && UpdateDepthTags(ref state);
            }

            state.Dependency = dependency;
        }

        [BurstCompile]
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static void RemoveAllDepthTags(in Entity entity, in EntityManager entityManager, in EntityCommandBuffer ecb)
        {
            if (entityManager.HasComponent<TransformDepth1>(entity)) ecb.RemoveComponent<TransformDepth1>(entity);
            if (entityManager.HasComponent<TransformDepth2>(entity)) ecb.RemoveComponent<TransformDepth2>(entity);
            if (entityManager.HasComponent<TransformDepth3>(entity)) ecb.RemoveComponent<TransformDepth3>(entity);
            if (entityManager.HasComponent<TransformDepth4>(entity)) ecb.RemoveComponent<TransformDepth4>(entity);
            if (entityManager.HasComponent<TransformDepth5>(entity)) ecb.RemoveComponent<TransformDepth5>(entity);
            if (entityManager.HasComponent<TransformDepth6>(entity)) ecb.RemoveComponent<TransformDepth6>(entity);
            if (entityManager.HasComponent<TransformDepth7>(entity)) ecb.RemoveComponent<TransformDepth7>(entity);
            if (entityManager.HasComponent<TransformDepth8>(entity)) ecb.RemoveComponent<TransformDepth8>(entity);
            if (entityManager.HasComponent<TransformDepthDeep>(entity)) ecb.RemoveComponent<TransformDepthDeep>(entity);
        }

        /// <summary>Returns whether children were marked dirty for the next update.</summary>
        [BurstCompile]
        private bool UpdateDepthTags(ref SystemState state)
        {
            var ecb = new EntityCommandBuffer(Allocator.TempJob);
            bool childrenMarked = false;

            var parentLookup = SystemAPI.GetComponentLookup<Parent>(true);
            var depth1Lookup = SystemAPI.GetComponentLookup<TransformDepth1>(true);
            var depth2Lookup = SystemAPI.GetComponentLookup<TransformDepth2>(true);
            var depth3Lookup = SystemAPI.GetComponentLookup<TransformDepth3>(true);
            var depth4Lookup = SystemAPI.GetComponentLookup<TransformDepth4>(true);
            var depth5Lookup = SystemAPI.GetComponentLookup<TransformDepth5>(true);
            var depth6Lookup = SystemAPI.GetComponentLookup<TransformDepth6>(true);
            var depth7Lookup = SystemAPI.GetComponentLookup<TransformDepth7>(true);
            var depth8Lookup = SystemAPI.GetComponentLookup<TransformDepth8>(true);
            var depthDeepLookup = SystemAPI.GetComponentLookup<TransformDepthDeep>(true);
            var childLookup = SystemAPI.GetBufferLookup<Child>(true);

            foreach (var (parent, entity) in
                SystemAPI.Query<RefRO<Parent>>()
                .WithAll<DepthDirty>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities)
                .WithEntityAccess())
            {
                int depth = CalculateDepth(parent.ValueRO.Value, parentLookup);

                RemoveAllDepthTags(entity, state.EntityManager, ecb);

                switch (depth)
                {
                    case 1: ecb.AddComponent<TransformDepth1>(entity); break;
                    case 2: ecb.AddComponent<TransformDepth2>(entity); break;
                    case 3: ecb.AddComponent<TransformDepth3>(entity); break;
                    case 4: ecb.AddComponent<TransformDepth4>(entity); break;
                    case 5: ecb.AddComponent<TransformDepth5>(entity); break;
                    case 6: ecb.AddComponent<TransformDepth6>(entity); break;
                    case 7: ecb.AddComponent<TransformDepth7>(entity); break;
                    case 8: ecb.AddComponent<TransformDepth8>(entity); break;
                    default: ecb.AddComponent<TransformDepthDeep>(entity); break;
                }

                if (childLookup.TryGetBuffer(entity, out var children))
                {
                    childrenMarked |= children.Length > 0;
                    MarkChildrenDirty(children, childLookup, ecb);
                }

                ecb.SetComponentEnabled<DepthDirty>(entity, false);
            }

            ecb.Playback(state.EntityManager);
            ecb.Dispose();
            return childrenMarked;
        }

        [BurstCompile]
        private static int CalculateDepth(in Entity parent, in ComponentLookup<Parent> parentLookup)
        {
            int depth = 1;
            var current = parent;
            int guard = 0;

            while (parentLookup.TryGetComponent(current, out var next))
            {
                depth++;
                if (depth >= 9)
                    return 9;

                current = next.Value;

                if (++guard > 100)
                    return 9;
            }

            return depth;
        }

        private static void MarkChildrenDirty(DynamicBuffer<Child> children,
            BufferLookup<Child> childLookup, EntityCommandBuffer ecb)
        {
            for (int i = 0; i < children.Length; i++)
            {
                var child = children[i].Value;
                ecb.SetComponentEnabled<DepthDirty>(child, true);

                if (childLookup.TryGetBuffer(child, out var grandchildren))
                {
                    MarkChildrenDirty(grandchildren, childLookup, ecb);
                }
            }
        }
    }

    // ============================================================================
    // MAIN LOCALTOWORLDSYSTEM
    // ============================================================================
    //
    // Two independent update paths:
    //
    //   ENABLED entities: Every frame, all depths 1-8 sequentially via
    //   UpdateDepthLevelJob (with change filtering). Deep entities (9+) every
    //   frame via recursive traversal from enabled depth-8 roots. Fast path.
    //
    //   DISABLED entities: One depth slot per frame in round-robin via
    //   UpdateDisabledDepthLevelJob. A chunk is skipped only when nothing it
    //   reads (LocalTransform, PostTransformMatrix, Parent, parent L2W), its own
    //   L2W and its order changed since that slot's last run, i.e. when the
    //   recompute would write the value already stored.
    //   Over DISABLED_SLOT_COUNT frames every disabled entity gets updated once.
    //   Slot 0-7 = depth 1-8, slot 8 = deep (9+) via recursive traversal from
    //   all depth-8 roots (incl disabled).
    //
    //   The disabled job runs on the "all" query (enabled + disabled) but skips
    //   chunks without the Disabled tag, so enabled entities are never
    //   double-processed.
    //
    // ============================================================================

    [UpdateInGroup(typeof(TransformSystemGroup))]
    [UpdateAfter(typeof(ParentSystem))]
    [BurstCompile]
    public partial struct UltraFastLocalToWorldSystem : ISystem
    {
        // Slots 0-7 = depth 1-8, slot 8 = deep (9+).
        // One slot per frame → full cycle = 9 frames.
        const int DISABLED_SLOT_COUNT = 9;

        private EntityQuery _rootQuery;

        // Enabled-only queries (every frame, change-filtered)
        private EntityQuery _depth1EnabledQuery;
        private EntityQuery _depth2EnabledQuery;
        private EntityQuery _depth3EnabledQuery;
        private EntityQuery _depth4EnabledQuery;
        private EntityQuery _depth5EnabledQuery;
        private EntityQuery _depth6EnabledQuery;
        private EntityQuery _depth7EnabledQuery;
        private EntityQuery _depth8EnabledQuery;

        // Disabled-only queries — one depth per frame, no change filter
        private EntityQuery _depth1DisabledQuery;
        private EntityQuery _depth2DisabledQuery;
        private EntityQuery _depth3DisabledQuery;
        private EntityQuery _depth4DisabledQuery;
        private EntityQuery _depth5DisabledQuery;
        private EntityQuery _depth6DisabledQuery;
        private EntityQuery _depth7DisabledQuery;
        private EntityQuery _depth8DisabledQuery;

        // Deep hierarchy — depth-8 entities with children, used as roots for recursive traversal
        private EntityQuery _depth8WithChildrenEnabledQuery;
        private EntityQuery _depth8WithChildrenDisabledQuery;
        private EntityQuery _markedHierarchyRebuildQuery;

        private EntityQueryMask _localToWorldWriteGroupMask;
        private EntityQueryMask _rootLocalToWorldWriteGroupMask;

        private ComponentTypeHandle<LocalTransform> _localTransformHandle;
        private ComponentTypeHandle<PostTransformMatrix> _postTransformHandle;
        private ComponentTypeHandle<LocalToWorld> _localToWorldHandle;
        private ComponentTypeHandle<Parent> _parentHandle;
        private EntityTypeHandle _entityHandle;

        private ComponentLookup<Parent> _parentLookup;
        private ComponentLookup<LocalTransform> _localTransformLookup;
        private ComponentLookup<PostTransformMatrix> _postTransformLookup;
        private ComponentLookup<LocalToWorld> _localToWorldLookup;
        private BufferLookup<Child> _childLookup;

        int _disabledSlot;

        // Global system version of the last update that ran each disabled depth slot (0-7), 0 before the first run.
        FixedList64Bytes<uint> _disabledSlotVersions;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            // Root entities (no parent)
            _rootQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld>()
                .WithNone<Parent>()
                .WithNone<PhysicsGraphicalSmoothing>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));

            // Enabled-only queries
            _depth1EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth1>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth2EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth2>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth3EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth3>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth4EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth4>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth5EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth5>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth6EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth6>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth7EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth7>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));
            _depth8EnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth8>()
                .WithOptions(EntityQueryOptions.FilterWriteGroup));

            // Disabled-only queries
            _depth1DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth1, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth2DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth2, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth3DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth3, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth4DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth4, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth5DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth5, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth6DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth6, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth7DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth7, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _depth8DisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld, Parent, TransformDepth8, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));

            // Depth-8 entities with children — roots for deep recursive traversal (IJobChunk)
            _depth8WithChildrenEnabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<TransformDepth8, Child, LocalToWorld>());
            _depth8WithChildrenDisabledQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<TransformDepth8, Child, LocalToWorld, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities));

            _markedHierarchyRebuildQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<DynamicLinkedEntityGroupSystem.HierarchyChanged, LocalTransform, LocalToWorld, Disabled>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities));

            var rootWriteGroupQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, LocalToWorld>()
                .WithNone<Parent>()
                .WithNone<PhysicsGraphicalSmoothing>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _rootLocalToWorldWriteGroupMask = rootWriteGroupQuery.GetEntityQueryMask();

            var writeGroupQuery = state.GetEntityQuery(new EntityQueryBuilder(Allocator.Temp)
                .WithAll<LocalTransform, Parent, LocalToWorld>()
                .WithOptions(EntityQueryOptions.IncludeDisabledEntities | EntityQueryOptions.FilterWriteGroup));
            _localToWorldWriteGroupMask = writeGroupQuery.GetEntityQueryMask();

            state.RequireForUpdate<Child>();

            _disabledSlot = 0;

            _disabledSlotVersions.Clear();
            for (int i = 0; i < 8; i++)
            {
                _disabledSlotVersions.Add(0);
            }
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            _localTransformHandle = SystemAPI.GetComponentTypeHandle<LocalTransform>(true);
            _postTransformHandle = SystemAPI.GetComponentTypeHandle<PostTransformMatrix>(true);
            _localToWorldHandle = SystemAPI.GetComponentTypeHandle<LocalToWorld>(false);
            _parentHandle = SystemAPI.GetComponentTypeHandle<Parent>(true);
            _entityHandle = SystemAPI.GetEntityTypeHandle();

            _parentLookup = SystemAPI.GetComponentLookup<Parent>(true);
            _localTransformLookup = SystemAPI.GetComponentLookup<LocalTransform>(true);
            _postTransformLookup = SystemAPI.GetComponentLookup<PostTransformMatrix>(true);
            _localToWorldLookup = SystemAPI.GetComponentLookup<LocalToWorld>(false);
            _childLookup = SystemAPI.GetBufferLookup<Child>(true);

            var dependency = state.Dependency;

            // ----------------------------------------------------------------
            // Step 1: Roots (no parent) — always, incl disabled, change-filtered
            // ----------------------------------------------------------------
            if (!_rootQuery.IsEmpty)
            {
                var rootJob = new UpdateRootJob
                {
                    LocalTransformHandle = _localTransformHandle,
                    PostTransformHandle = _postTransformHandle,
                    LocalToWorldHandle = _localToWorldHandle,
                    LastSystemVersion = state.LastSystemVersion
                };
                dependency = rootJob.ScheduleParallel(_rootQuery, dependency);
            }

            // ----------------------------------------------------------------
            // Step 2: ENABLED entities — every frame, all depths, change-filtered
            // ----------------------------------------------------------------
            {
                var enabledJob = new UpdateDepthLevelJob
                {
                    LocalTransformHandle = _localTransformHandle,
                    PostTransformHandle = _postTransformHandle,
                    LocalToWorldHandle = _localToWorldHandle,
                    EntityHandle = _entityHandle,
                    ParentLookup = _parentLookup,
                    ParentHandle = _parentHandle,
                    ParentLocalToWorldLookup = _localToWorldLookup,
                    PostTransformLookup = _postTransformLookup,
                    LastSystemVersion = state.LastSystemVersion
                };

                if (!_depth1EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth1EnabledQuery, dependency);
                if (!_depth2EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth2EnabledQuery, dependency);
                if (!_depth3EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth3EnabledQuery, dependency);
                if (!_depth4EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth4EnabledQuery, dependency);
                if (!_depth5EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth5EnabledQuery, dependency);
                if (!_depth6EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth6EnabledQuery, dependency);
                if (!_depth7EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth7EnabledQuery, dependency);
                if (!_depth8EnabledQuery.IsEmpty) dependency = enabledJob.ScheduleParallel(_depth8EnabledQuery, dependency);
            }

            // ----------------------------------------------------------------
            // Step 3: Enabled deep entities (depth 9+) — every frame, recursive
            //         via IJobChunk on depth-8 roots that have children
            // ----------------------------------------------------------------
            if (!_depth8WithChildrenEnabledQuery.IsEmpty)
            {
                var deepJob = new UpdateDeepHierarchyJob
                {
                    ChildLookup = _childLookup,
                    LocalTransformLookup = _localTransformLookup,
                    PostTransformLookup = _postTransformLookup,
                    LocalToWorldLookup = _localToWorldLookup,
                    LocalToWorldWriteGroupMask = _localToWorldWriteGroupMask,
                    EntityHandle = _entityHandle,
                    LocalToWorldHandle = _localToWorldHandle,
                };
                dependency = deepJob.ScheduleParallel(_depth8WithChildrenEnabledQuery, dependency);
            }

            // ----------------------------------------------------------------
            // Step 4: DISABLED entities — one depth slot per frame, skipping chunks unchanged since the slot's last run
            // ----------------------------------------------------------------
            if (_disabledSlot < 8)
            {
                var disabledQuery = _disabledSlot switch
                {
                    0 => _depth1DisabledQuery,
                    1 => _depth2DisabledQuery,
                    2 => _depth3DisabledQuery,
                    3 => _depth4DisabledQuery,
                    4 => _depth5DisabledQuery,
                    5 => _depth6DisabledQuery,
                    6 => _depth7DisabledQuery,
                    _ => _depth8DisabledQuery,
                };

                if (!disabledQuery.IsEmpty)
                {
                    var disabledJob = new UpdateDisabledDepthLevelJob
                    {
                        LocalTransformHandle = _localTransformHandle,
                        PostTransformHandle = _postTransformHandle,
                        ParentHandle = _parentHandle,
                        LocalToWorldHandle = _localToWorldHandle,
                        EntityHandle = _entityHandle,
                        ParentLocalToWorldLookup = _localToWorldLookup,
                        PostTransformLookup = _postTransformLookup,
                        LastSlotVersion = _disabledSlotVersions[_disabledSlot],
                    };
                    dependency = disabledJob.ScheduleParallel(disabledQuery, dependency);
                    _disabledSlotVersions[_disabledSlot] = state.GlobalSystemVersion;
                }
            }
            else
            {
                // Slot 8: deep hierarchy (depth 9+) — recursive from disabled depth-8 roots
                if (!_depth8WithChildrenDisabledQuery.IsEmpty)
                {
                    var deepJob = new UpdateDeepHierarchyJob
                    {
                        ChildLookup = _childLookup,
                        LocalTransformLookup = _localTransformLookup,
                        PostTransformLookup = _postTransformLookup,
                        LocalToWorldLookup = _localToWorldLookup,
                        LocalToWorldWriteGroupMask = _localToWorldWriteGroupMask,
                        EntityHandle = _entityHandle,
                        LocalToWorldHandle = _localToWorldHandle,
                    };
                    dependency = deepJob.ScheduleParallel(_depth8WithChildrenDisabledQuery, dependency);
                }
            }

            if (!_markedHierarchyRebuildQuery.IsEmpty)
            {
                var markedHierarchyRebuildJob = new RebuildMarkedHierarchyJob
                {
                    EntityHandle = _entityHandle,
                    ParentLookup = _parentLookup,
                    ChildLookup = _childLookup,
                    LocalTransformLookup = _localTransformLookup,
                    PostTransformLookup = _postTransformLookup,
                    LocalToWorldWriteGroupMask = _localToWorldWriteGroupMask,
                    RootLocalToWorldWriteGroupMask = _rootLocalToWorldWriteGroupMask,
                    LocalToWorldLookup = _localToWorldLookup,
                };
                dependency = markedHierarchyRebuildJob.Schedule(_markedHierarchyRebuildQuery, dependency);

                // It writes after this update's disabled slot job with the same version, which that slot's change check
                // cannot see, so the slot recomputes every chunk on its next run.
                if (_disabledSlot < 8)
                {
                    _disabledSlotVersions[_disabledSlot] = 0;
                }
            }

            state.Dependency = dependency;

            _disabledSlot = (_disabledSlot + 1) % DISABLED_SLOT_COUNT;
        }

        // ====================================================================
        // JOBS
        // ====================================================================

        /// <summary>
        /// Root entities (no parent). Change-filtered.
        /// </summary>
        [BurstCompile]
        private struct UpdateRootJob : IJobChunk
        {
            [ReadOnly] public ComponentTypeHandle<LocalTransform> LocalTransformHandle;
            [ReadOnly] public ComponentTypeHandle<PostTransformMatrix> PostTransformHandle;
            public ComponentTypeHandle<LocalToWorld> LocalToWorldHandle;
            public uint LastSystemVersion;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                if (!chunk.DidChange(ref LocalTransformHandle, LastSystemVersion) &&
                    !(chunk.Has(ref PostTransformHandle) && chunk.DidChange(ref PostTransformHandle, LastSystemVersion)))
                    return;

                var localTransforms = chunk.GetNativeArray(ref LocalTransformHandle);
                var localToWorlds = chunk.GetNativeArray(ref LocalToWorldHandle);

                if (chunk.Has(ref PostTransformHandle))
                {
                    var postTransforms = chunk.GetNativeArray(ref PostTransformHandle);
                    for (int i = 0; i < chunk.Count; i++)
                    {
                        var matrix = math.mul(localTransforms[i].ToMatrix(), postTransforms[i].Value);
                        localToWorlds[i] = new LocalToWorld { Value = matrix };
                    }
                }
                else
                {
                    for (int i = 0; i < chunk.Count; i++)
                    {
                        var matrix = localTransforms[i].ToMatrix();
                        localToWorlds[i] = new LocalToWorld { Value = matrix };
                    }
                }
            }
        }

        /// <summary>
        /// Enabled child entities. Uses change filtering to skip unchanged chunks.
        /// </summary>
        [BurstCompile]
        private struct UpdateDepthLevelJob : IJobChunk
        {
            [ReadOnly] public ComponentTypeHandle<LocalTransform> LocalTransformHandle;
            [ReadOnly] public ComponentTypeHandle<PostTransformMatrix> PostTransformHandle;
            [ReadOnly] public ComponentTypeHandle<Parent> ParentHandle;

            [NativeDisableContainerSafetyRestriction]
            public ComponentTypeHandle<LocalToWorld> LocalToWorldHandle;

            [ReadOnly] public EntityTypeHandle EntityHandle;
            [ReadOnly] public ComponentLookup<Parent> ParentLookup;
            [ReadOnly] public ComponentLookup<LocalToWorld> ParentLocalToWorldLookup;
            [ReadOnly] public ComponentLookup<PostTransformMatrix> PostTransformLookup;
            public uint LastSystemVersion;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                bool localTransformChanged = chunk.DidChange(ref LocalTransformHandle, LastSystemVersion);
                bool postTransformChanged = chunk.Has(ref PostTransformHandle) && chunk.DidChange(ref PostTransformHandle, LastSystemVersion);
                bool parentChanged = chunk.DidChange(ref ParentHandle, LastSystemVersion);
                var parents = chunk.GetNativeArray(ref ParentHandle);

                if (!localTransformChanged && !postTransformChanged && !parentChanged)
                {
                    bool anyParentL2WChanged = false;
                    for (int i = 0; i < chunk.Count; i++)
                    {
                        var parent = parents[i];
                        if (ParentLocalToWorldLookup.HasComponent(parent.Value) &&
                            ParentLocalToWorldLookup.DidChange(parent.Value, LastSystemVersion))
                        {
                            anyParentL2WChanged = true;
                            break;
                        }
                    }
                    if (!anyParentL2WChanged)
                        return;
                }

                var localTransforms = chunk.GetNativeArray(ref LocalTransformHandle);
                var localToWorlds = chunk.GetNativeArray(ref LocalToWorldHandle);
                var hasPostTransform = chunk.Has(ref PostTransformHandle);

                NativeArray<PostTransformMatrix> postTransforms = default;
                if (hasPostTransform)
                    postTransforms = chunk.GetNativeArray(ref PostTransformHandle);

                for (int i = 0; i < chunk.Count; i++)
                {
                    var parent = parents[i];

                    if (!ParentLocalToWorldLookup.TryGetComponent(parent.Value, out var parentL2W))
                        continue;

                    float4x4 localMatrix = localTransforms[i].ToMatrix();

                    if (hasPostTransform)
                        localMatrix = math.mul(localMatrix, postTransforms[i].Value);

                    var finalMatrix = math.mul(parentL2W.Value, localMatrix);
                    localToWorlds[i] = new LocalToWorld { Value = finalMatrix };
                }
            }
        }

        /// <summary>
        /// Disabled child entities, scheduled on one depth level per frame in round-robin.
        /// A chunk is recomputed only when something it reads, or its own LocalToWorld, was written after
        /// the previous run of its slot, so a skipped chunk already holds what the recompute would write.
        /// </summary>
        [BurstCompile]
        private struct UpdateDisabledDepthLevelJob : IJobChunk
        {
            [ReadOnly] public ComponentTypeHandle<LocalTransform> LocalTransformHandle;
            [ReadOnly] public ComponentTypeHandle<PostTransformMatrix> PostTransformHandle;
            [ReadOnly] public ComponentTypeHandle<Parent> ParentHandle;

            [NativeDisableContainerSafetyRestriction]
            public ComponentTypeHandle<LocalToWorld> LocalToWorldHandle;

            [ReadOnly] public EntityTypeHandle EntityHandle;
            [ReadOnly] public ComponentLookup<LocalToWorld> ParentLocalToWorldLookup;
            [ReadOnly] public ComponentLookup<PostTransformMatrix> PostTransformLookup;

            /// <summary>
            /// Global system version of the previous run of this slot, which is also the version of that run's writes; 0 before the first run.
            /// </summary>
            public uint LastSlotVersion;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                // Must run before LocalToWorld is fetched for writing below, which marks the chunk changed.
                if (!NeedsUpdate(chunk))
                {
                    return;
                }

                var localTransforms = chunk.GetNativeArray(ref LocalTransformHandle);
                var localToWorlds = chunk.GetNativeArray(ref LocalToWorldHandle);
                var parents = chunk.GetNativeArray(ref ParentHandle);
                var hasPostTransform = chunk.Has(ref PostTransformHandle);

                NativeArray<PostTransformMatrix> postTransforms = default;
                if (hasPostTransform)
                    postTransforms = chunk.GetNativeArray(ref PostTransformHandle);

                for (int i = 0; i < chunk.Count; i++)
                {
                    var parent = parents[i];

                    if (!ParentLocalToWorldLookup.TryGetComponent(parent.Value, out var parentL2W))
                        continue;

                    float4x4 localMatrix = localTransforms[i].ToMatrix();

                    if (hasPostTransform)
                        localMatrix = math.mul(localMatrix, postTransforms[i].Value);

                    var finalMatrix = math.mul(parentL2W.Value, localMatrix);
                    localToWorlds[i] = new LocalToWorld { Value = finalMatrix };
                }
            }

            bool NeedsUpdate(in ArchetypeChunk chunk)
            {
                // Own LocalToWorld counts so that writes by anything else are overwritten as before; this job's own writes carry LastSlotVersion.
                if (chunk.DidOrderChange(LastSlotVersion)
                    || chunk.DidChange(ref LocalTransformHandle, LastSlotVersion)
                    || chunk.DidChange(ref ParentHandle, LastSlotVersion)
                    || chunk.DidChange(ref LocalToWorldHandle, LastSlotVersion)
                    || (chunk.Has(ref PostTransformHandle) && chunk.DidChange(ref PostTransformHandle, LastSlotVersion)))
                {
                    return true;
                }

                // Parents are compared one version earlier, so a parent written with the previous run's own version also counts
                // (a parent in the same slot while its depth tag is stale). Parents are a level up, so normal runs never retrigger themselves.
                uint parentSinceVersion = LastSlotVersion - 1;
                var parents = chunk.GetNativeArray(ref ParentHandle);
                var checkedParent = Entity.Null;
                for (int i = 0; i < chunk.Count; i++)
                {
                    var parent = parents[i].Value;
                    if (parent == checkedParent)
                    {
                        continue;
                    }

                    checkedParent = parent;

                    if (ParentLocalToWorldLookup.HasComponent(parent) && ParentLocalToWorldLookup.DidChange(parent, parentSinceVersion))
                    {
                        return true;
                    }
                }

                return false;
            }
        }

        /// <summary>
        /// Immediate full rebuild for hierarchies whose structure is about to be
        /// consumed by gameplay systems. This bypasses the disabled round-robin
        /// path so sockets and dynamic linked groups see current LocalToWorld.
        /// </summary>
        [BurstCompile]
        private struct RebuildMarkedHierarchyJob : IJobChunk
        {
            [ReadOnly] public EntityTypeHandle EntityHandle;
            [ReadOnly] public ComponentLookup<Parent> ParentLookup;
            [ReadOnly] public BufferLookup<Child> ChildLookup;
            [ReadOnly] public ComponentLookup<LocalTransform> LocalTransformLookup;
            [ReadOnly] public ComponentLookup<PostTransformMatrix> PostTransformLookup;
            [ReadOnly] public EntityQueryMask LocalToWorldWriteGroupMask;
            [ReadOnly] public EntityQueryMask RootLocalToWorldWriteGroupMask;

            [NativeDisableParallelForRestriction][NativeDisableContainerSafetyRestriction]
            public ComponentLookup<LocalToWorld> LocalToWorldLookup;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                var entities = chunk.GetNativeArray(EntityHandle);

                for (int i = 0; i < chunk.Count; i++)
                {
                    var rootEntity = entities[i];
                    if (!TryBuildWorldFromCurrentHierarchy(rootEntity, out var rootLocalToWorld, out var canWriteRootLocalToWorld, 0))
                        continue;

                    if (canWriteRootLocalToWorld)
                        LocalToWorldLookup[rootEntity] = rootLocalToWorld;

                    if (ChildLookup.TryGetBuffer(rootEntity, out var children))
                    {
                        RebuildChildrenRecursive(children, rootLocalToWorld);
                    }
                }
            }

            private bool TryBuildWorldFromCurrentHierarchy(Entity entity, out LocalToWorld localToWorld, out bool canWriteLocalToWorld, int depth)
            {
                localToWorld = default;
                canWriteLocalToWorld = false;

                if (depth > 128)
                    return false;

                if (!LocalTransformLookup.TryGetComponent(entity, out var localTransform))
                    return LocalToWorldLookup.TryGetComponent(entity, out localToWorld);

                var localMatrix = BuildLocalMatrix(entity, localTransform);
                if (ParentLookup.TryGetComponent(entity, out var parent) &&
                    TryBuildWorldFromCurrentHierarchy(parent.Value, out var parentLocalToWorld, out _, depth + 1))
                {
                    canWriteLocalToWorld = CanWriteLocalToWorld(entity);
                    if (!canWriteLocalToWorld)
                        return LocalToWorldLookup.TryGetComponent(entity, out localToWorld);

                    localToWorld = new LocalToWorld { Value = math.mul(parentLocalToWorld.Value, localMatrix) };
                    return true;
                }

                canWriteLocalToWorld = CanWriteLocalToWorld(entity);
                if (!canWriteLocalToWorld)
                    return LocalToWorldLookup.TryGetComponent(entity, out localToWorld);

                localToWorld = new LocalToWorld { Value = localMatrix };
                return true;
            }

            private void RebuildChildrenRecursive(DynamicBuffer<Child> children, LocalToWorld parentLocalToWorld)
            {
                for (int i = 0; i < children.Length; i++)
                {
                    var childEntity = children[i].Value;
                    if (!TryBuildChildLocalToWorld(childEntity, parentLocalToWorld, out var childLocalToWorld))
                        continue;

                    if (ChildLookup.TryGetBuffer(childEntity, out var grandchildren))
                    {
                        RebuildChildrenRecursive(grandchildren, childLocalToWorld);
                    }
                }
            }

            private bool TryBuildChildLocalToWorld(Entity entity, LocalToWorld parentLocalToWorld, out LocalToWorld localToWorld)
            {
                localToWorld = default;

                if (!LocalTransformLookup.TryGetComponent(entity, out var localTransform))
                    return LocalToWorldLookup.TryGetComponent(entity, out localToWorld);

                if (!CanWriteLocalToWorld(entity))
                    return LocalToWorldLookup.TryGetComponent(entity, out localToWorld);

                localToWorld = new LocalToWorld
                {
                    Value = math.mul(parentLocalToWorld.Value, BuildLocalMatrix(entity, localTransform))
                };

                LocalToWorldLookup[entity] = localToWorld;
                return true;
            }

            private bool CanWriteLocalToWorld(Entity entity)
            {
                return ParentLookup.HasComponent(entity)
                    ? LocalToWorldWriteGroupMask.MatchesIgnoreFilter(entity)
                    : RootLocalToWorldWriteGroupMask.MatchesIgnoreFilter(entity);
            }

            private float4x4 BuildLocalMatrix(Entity entity, LocalTransform localTransform)
            {
                var localMatrix = localTransform.ToMatrix();
                if (PostTransformLookup.TryGetComponent(entity, out var postTransform))
                    localMatrix = math.mul(localMatrix, postTransform.Value);

                return localMatrix;
            }
        }

        /// <summary>
        /// Recursive traversal for depth 9+ entities. Runs as IJobChunk on
        /// depth-8 entities that have children. Each chunk entity is a root
        /// from which children are recursively updated.
        /// </summary>
        [BurstCompile]
        private struct UpdateDeepHierarchyJob : IJobChunk
        {
            [ReadOnly] public EntityTypeHandle EntityHandle;
            [ReadOnly][NativeDisableContainerSafetyRestriction] public ComponentTypeHandle<LocalToWorld> LocalToWorldHandle;
            [ReadOnly] public BufferLookup<Child> ChildLookup;
            [ReadOnly] public ComponentLookup<LocalTransform> LocalTransformLookup;
            [ReadOnly] public ComponentLookup<PostTransformMatrix> PostTransformLookup;
            [NativeDisableParallelForRestriction][NativeDisableContainerSafetyRestriction] public ComponentLookup<LocalToWorld> LocalToWorldLookup;

            [ReadOnly] public EntityQueryMask LocalToWorldWriteGroupMask;

            public void Execute(in ArchetypeChunk chunk, int unfilteredChunkIndex, bool useEnabledMask, in v128 chunkEnabledMask)
            {
                var entities = chunk.GetNativeArray(EntityHandle);
                var localToWorlds = chunk.GetNativeArray(ref LocalToWorldHandle);

                for (int i = 0; i < chunk.Count; i++)
                {
                    var rootEntity = entities[i];
                    var rootL2W = localToWorlds[i];

                    if (ChildLookup.TryGetBuffer(rootEntity, out var children))
                    {
                        UpdateChildrenRecursive(children, rootL2W);
                    }
                }
            }

            private void UpdateChildrenRecursive(DynamicBuffer<Child> children, LocalToWorld parentL2W)
            {
                for (int i = 0; i < children.Length; i++)
                {
                    var childEntity = children[i].Value;

                    if (!LocalToWorldWriteGroupMask.MatchesIgnoreFilter(childEntity))
                        continue;

                    if (!LocalTransformLookup.TryGetComponent(childEntity, out var localTransform))
                        continue;

                    float4x4 localMatrix = localTransform.ToMatrix();

                    if (PostTransformLookup.TryGetComponent(childEntity, out var postTransform))
                        localMatrix = math.mul(localMatrix, postTransform.Value);

                    var finalMatrix = math.mul(parentL2W.Value, localMatrix);
                    var childL2W = new LocalToWorld { Value = finalMatrix };
                    LocalToWorldLookup[childEntity] = childL2W;

                    if (ChildLookup.TryGetBuffer(childEntity, out var grandchildren))
                    {
                        UpdateChildrenRecursive(grandchildren, childL2W);
                    }
                }
            }
        }
    }
}
