-- ============================================================================
-- Daily Segment Breakdown (for the Diagnose step)
-- Grain: date x device_category x medium
-- This is what the drill-down layer queries when a daily KPI anomaly fires,
-- to compute each segment's contribution to the overall change.
-- ============================================================================

WITH events AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS event_date,
    user_pseudo_id,
    event_name,
    device.category                  AS device_category,
    traffic_source.medium            AS medium,
    traffic_source.source            AS source,
    ecommerce.purchase_revenue       AS purchase_revenue
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
)

SELECT
  event_date,
  COALESCE(device_category, '(not set)')                                AS device_category,
  COALESCE(medium, '(none)')                                            AS medium,
  COALESCE(source, '(direct)')                                          AS source,
  COUNT(DISTINCT user_pseudo_id)                                        AS users,
  COUNT(DISTINCT IF(event_name = 'purchase', user_pseudo_id, NULL))     AS purchasers,
  ROUND(SAFE_DIVIDE(
    COUNT(DISTINCT IF(event_name = 'purchase', user_pseudo_id, NULL)),
    COUNT(DISTINCT user_pseudo_id)), 4)                                 AS conversion_rate,
  ROUND(SUM(IF(event_name = 'purchase', purchase_revenue, 0)), 2)       AS revenue
FROM events
GROUP BY event_date, device_category, medium, source
ORDER BY event_date, users DESC;
