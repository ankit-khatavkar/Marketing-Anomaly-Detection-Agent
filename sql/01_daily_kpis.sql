-- ============================================================================
-- Daily Marketing KPI Extraction
-- Source: bigquery-public-data.ga4_obfuscated_sample_ecommerce
-- Output: one row per calendar day with core marketing KPIs
-- ============================================================================

WITH events AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date)                              AS event_date,
    user_pseudo_id,
    event_name,
    device.category                                                AS device_category,
    traffic_source.medium                                          AS medium,
    traffic_source.source                                          AS source,
    (SELECT value.int_value
       FROM UNNEST(event_params)
      WHERE key = 'ga_session_id')                                 AS session_id,
    (SELECT value.string_value
       FROM UNNEST(event_params)
      WHERE key = 'session_engaged')                                AS session_engaged,
    (SELECT value.int_value
       FROM UNNEST(event_params)
      WHERE key = 'engagement_time_msec')                          AS engagement_time_msec,
    ecommerce.purchase_revenue                                     AS purchase_revenue
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
),

daily_sessions AS (
  SELECT
    event_date,
    CONCAT(user_pseudo_id, '-', CAST(session_id AS STRING)) AS session_key,
    MAX(IF(session_engaged = '1', 1, 0))                    AS is_engaged,
    SUM(engagement_time_msec)                               AS engagement_time_msec,
    ANY_VALUE(device_category)                              AS device_category,
    ANY_VALUE(medium)                                       AS medium
  FROM events
  WHERE session_id IS NOT NULL
  GROUP BY event_date, session_key
),

daily_agg AS (
  SELECT
    event_date,
    COUNT(DISTINCT user_pseudo_id)                                        AS total_users,
    COUNTIF(event_name = 'purchase')                                      AS purchase_events,
    COUNT(DISTINCT IF(event_name = 'purchase', user_pseudo_id, NULL))     AS purchasers,
    SUM(IF(event_name = 'purchase', purchase_revenue, 0))                 AS revenue,
    COUNTIF(event_name = 'session_start')                                 AS sessions_started
  FROM events
  GROUP BY event_date
),

daily_engagement AS (
  SELECT
    event_date,
    COUNT(*)                                        AS total_sessions,
    SUM(is_engaged)                                  AS engaged_sessions,
    SAFE_DIVIDE(SUM(is_engaged), COUNT(*))           AS engagement_rate
  FROM daily_sessions
  GROUP BY event_date
)

SELECT
  a.event_date,
  FORMAT_DATE('%A', a.event_date)                          AS day_of_week,
  a.total_users,
  a.sessions_started,
  e.total_sessions,
  e.engaged_sessions,
  ROUND(e.engagement_rate, 4)                              AS engagement_rate,
  a.purchase_events,
  a.purchasers,
  ROUND(SAFE_DIVIDE(a.purchasers, a.total_users), 4)       AS conversion_rate,
  ROUND(a.revenue, 2)                                      AS revenue,
  ROUND(SAFE_DIVIDE(a.revenue, a.purchasers), 2)           AS avg_order_value
FROM daily_agg a
JOIN daily_engagement e USING (event_date)
ORDER BY a.event_date;
