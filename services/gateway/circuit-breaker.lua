local _M = {}
local shared = ngx.shared.circuit_breakers

local FAILURE_THRESHOLD = 5
local SUCCESS_THRESHOLD = 2
local TIMEOUT = 30
local HALF_OPEN_REQUESTS = 3

function _M.init()
    -- Initialize circuit breakers
end

function _M.check(service_name)
    local key = "cb:" .. service_name
    local state = shared:get(key .. ":state") or "closed"
    
    if state == "open" then
        local open_time = shared:get(key .. ":open_time") or 0
        if ngx.now() - open_time > TIMEOUT then
            shared:set(key .. ":state", "half_open")
            shared:set(key .. ":half_open_requests", 0)
        else
            ngx.status = 503
            ngx.say('{"error":"Service temporarily unavailable"}')
            ngx.exit(503)
        end
    elseif state == "half_open" then
        local requests = shared:incr(key .. ":half_open_requests", 1) or 1
        if requests > HALF_OPEN_REQUESTS then
            ngx.status = 503
            ngx.say('{"error":"Service in recovery"}')
            ngx.exit(503)
        end
    end
end

function _M.record(status)
    local service_name = ngx.var.upstream
    local key = "cb:" .. service_name
    
    if not status then
        status = 0
    end
    
    if status >= 500 or status == 0 then
        local failures = shared:incr(key .. ":failures", 1) or 1
        shared:set(key .. ":successes", 0)
        
        if failures >= FAILURE_THRESHOLD then
            shared:set(key .. ":state", "open")
            shared:set(key .. ":open_time", ngx.now())
            shared:set(key .. ":failures", 0)
        end
    else
        local successes = shared:incr(key .. ":successes", 1) or 1
        local state = shared:get(key .. ":state") or "closed"
        
        if state == "half_open" and successes >= SUCCESS_THRESHOLD then
            shared:set(key .. ":state", "closed")
            shared:set(key .. ":failures", 0)
            shared:set(key .. ":successes", 0)
        end
    end
end

return _M