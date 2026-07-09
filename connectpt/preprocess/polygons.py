import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram
from shapely.ops import unary_union, voronoi_diagram 

from .utils import _remove_water_objects

def build_new_zones(
        points_gdf: gpd.GeoDataFrame,
        utm_crs: int | str,
        territory: gpd.GeoDataFrame,
        clusters_gdf : gpd.GeoDataFrame,
        buffer: int = 1000, 
        water_gdf: gpd.GeoDataFrame = None,
        remove_water: bool = False,
        clip_by_territory: bool = False,
        merge_to_zones: bool = True
        
) -> gpd.GeoDataFrame:
    # multipoint
    print("Preparing data...")
    points_gdf_copy = points_gdf.copy().to_crs(utm_crs)
    points_multipoint = MultiPoint(points_gdf_copy.geometry.tolist())
    territory_copy = territory.copy().to_crs(utm_crs)
    territory_copy = territory_copy.buffer(0)
    territory_pol = territory_copy.geometry.iloc[0]
    envelope = territory_copy.geometry.iloc[0].buffer(buffer)

    # build voronoi polygons
    print("Building voronoi diagram...")
    voronoi = voronoi_diagram(points_multipoint, envelope=envelope)
    voronoi_gdf = gpd.GeoDataFrame(geometry=[poly for poly in voronoi.geoms], crs=utm_crs)

    # water features
    if remove_water:
        print("Removing water objects...")
        if water_gdf is None:
            raise ValueError("water_gdf must be provided if remove_water is True")
        
        water_gdf_utm = water_gdf.to_crs(utm_crs)
        voronoi_gdf = _remove_water_objects(water_gdf_utm, voronoi_gdf)
        voronoi_gdf = voronoi_gdf.explode(index_parts=False).reset_index(drop=True)
    
    # clip by territory
    if clip_by_territory:
        print("Clipping territory by defined are...")
        clipped_list = []
        for idx, row in voronoi_gdf.iterrows():
            if row.geometry is None:
                continue
            clipped_geom = row.geometry.intersection(territory_pol).buffer(0)
            if not clipped_geom.is_empty:
                clipped_list.append(clipped_geom)
        
        voronoi_gdf = gpd.GeoDataFrame(geometry=clipped_list, crs=utm_crs)
        voronoi_gdf = voronoi_gdf.explode(index_parts=False).reset_index(drop=True)


    print("Comparing polygons and bus stops...")
    voronoi_gdf = (
        gpd.sjoin_nearest(voronoi_gdf, points_gdf_copy, how='left', distance_col='dist')
        .sort_values(['dist', 'index_right'])
        .drop_duplicates(subset='geometry', keep='first')
        .reset_index(drop=True)
    )

    # merge voronoi polygons to zones 
    merged_voronoi_list = []
    if merge_to_zones:
        print("Merging voronoi polygons to zones...")
        for cluster_id, group in voronoi_gdf.groupby('cluster_id'):
            merged_polygon = unary_union(group.geometry.tolist()) 
            clusters_gdf_slice_cluster_id = clusters_gdf[clusters_gdf.cluster_id == cluster_id]
            name_for_cluster = clusters_gdf_slice_cluster_id.name.iloc[0]

            merged_voronoi_list.append({ 
                'geometry': merged_polygon, 
                'cluster_id': cluster_id, 
                # 'name': group.iloc[0]['name'] 
                'name': name_for_cluster
            }) 
    
        voronoi_gdf = gpd.GeoDataFrame(merged_voronoi_list, crs=voronoi_gdf.crs)
    print("Done!")
    return voronoi_gdf
