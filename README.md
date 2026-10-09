# Unity Optimized Transforms

An alternative `LocalToWorld` updater for Unity Entities, aimed at large, wide transform hierarchies. It groups children by depth and processes chunks in parallel within each level, so work under a single large root can spread across worker threads.

Enabled hierarchies update every frame with change checks. Disabled children update one depth slot per frame over a nine-frame cycle. Levels 1-8 use chunk jobs; deeper descendants use recursive traversal. Depth tags are maintained when parents change.

## Use

Copy [OptimizedTransforms.cs](OptimizedTransforms.cs) and [Examples/DynamicLinkedEntityGroupSystem.cs](Examples/DynamicLinkedEntityGroupSystem.cs) into your project. The systems register automatically and disable Unity's built-in `LocalToWorldSystem` in their world. With a custom bootstrap, include the normal transform systems and these systems in the simulation world.

The implementation is shared unchanged. The example supplies its hierarchy-change tag and cleanup; it does not maintain linked entity groups or detect reparenting for you. If you already have that type, keep your own implementation instead of copying the example.

Used with Unity **6000.3.11f1** and these packages:

| Package | Version |
| --- | --- |
| Entities | 1.4.8 |
| Unity Physics | 1.4.7 |
| Burst | 1.8.30 |
| Collections | 2.6.8 |
| Mathematics | 1.3.3 |

Unity Physics is required by the `PhysicsGraphicalSmoothing` exclusion. For a custom assembly, reference `Unity.Entities`, `Unity.Transforms`, `Unity.Physics`, `Unity.Burst`, `Unity.Collections` and `Unity.Mathematics`, and enable unsafe code for generated Entities code. Other versions have not been checked.

## Refreshing a disabled hierarchy

A disabled hierarchy can carry stale `LocalToWorld` values between its scheduled updates. To refresh it on the next transform update, add `DynamicLinkedEntityGroupSystem.HierarchyChanged` to its root, which must have `Disabled`, `LocalTransform` and `LocalToWorld`.

For example, when reparenting between two disabled hierarchies:

```csharp
ecb.SetComponent(child, new Parent { Value = newParent });
ecb.AddComponent<DynamicLinkedEntityGroupSystem.HierarchyChanged>(oldRoot);
ecb.AddComponent<DynamicLinkedEntityGroupSystem.HierarchyChanged>(newRoot);
```

Play back those commands **before `TransformSystemGroup`** so `ParentSystem` updates the child buffers first. Read the rebuilt matrices after the transform group, observing normal job dependencies. The example schedules tag removal after that group and plays it back at the end of simulation. Ordinary movement needs no tag if delayed updates are acceptable.

## Current limitations

- The updater requires an enabled entity with a `Child` buffer. A world containing only roots, or only disabled hierarchies, can stop updating transforms after the built-in system is disabled.
- The queries currently declare `LocalToWorld` read-only when setting up `FilterWriteGroup`. Custom transform writers are therefore not reliably excluded and their results can be overwritten.
- Frequent reparenting adds structural work. Deep or mixed enabled/disabled hierarchies need particular care; depth-tag changes can take multiple frames to propagate.

The intended workload is large hierarchies with mostly stable parent relationships. Measure it with your own hierarchy sizes and update patterns; no comparative benchmark results are included.

## License

[MIT](LICENSE).
