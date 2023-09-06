local function read_inv_text(filename)
    -- read file
    local file = io.open(filename, "r")
    if file == nil then
        return nil
    end
    local str = file:read("a")
    file:close()


    local project = str:match("# Project: (%S+)")
    local version = str:match("# Version: (%S+)")

    local data = {project = project, version = version, items = {}}

    local ptn_data =
        "^" ..
        "(.-)%s+" ..        -- name
        "([%S:]-):" ..      -- domain
        "([%S]+)%s+" ..     -- role
        "(%-?%d+)%s+" ..     -- priority
        "(%S*)%s+" ..       -- uri
        "(.-)\r?$"         -- dispname


    -- Iterate through each line in the file content
    for line in str:gmatch("[^\r\n]+") do
        if not line:match("^#") then
            -- Match each line against the pattern
            local name, domain, role, priority, uri, dispName = line:match(ptn_data)

            -- if name is nil, raise an error
            if name == nil then
                error("Error parsing line: " .. line)
            end

            data.items[#data.items + 1] = {
                name = name,
                domain = domain,
                role = role,
                priority = priority,
                uri = uri,
                dispName = dispName
            }
        end
    end
    return data
end

local function read_json(filename)

    local file = io.open(filename, "r")
    if file == nil then
        return nil
    end
    local str = file:read("a")
    file:close()

    local decoded = quarto.json.decode(str)
    return decoded
end

local function read_inv_text_or_json(base_name)
    local file = io.open(base_name .. ".txt", "r")
    if file then
        -- TODO: refactors so we don't just close the file immediately
        io.close(file)
        json = read_inv_text(base_name .. ".txt")

    else
        json = read_json(base_name .. ".json")
    end

    return json
end

local inventory = {}

local function lookup(search_object)

    local results = {}
    for _, inv in ipairs(inventory) do
        for _, item in ipairs(inv.items) do
            -- e.g. :external+<inv_name>:<domain>:<role>:`<name>`
            if item.inv_name and item.inv_name ~= search_object.inv_name then
                goto continue
            end

            if item.name ~= search_object.name then
                goto continue
            end

            if search_object.role and item.role ~= search_object.role then
                goto continue
            end

            if search_object.domain and item.domain ~= search_object.domain then
                goto continue
            else
                if search_object.domain or item.domain == "py" then
                  table.insert(results, item)
                end

                goto continue
            end

            ::continue::
        end
    end

    if #results == 1 then
        return results[1]
    end
    if #results > 1 then
        quarto.log.warning("Found multiple matches for " .. search_object.name .. ", using the first match.")
        return results[1]
    end
    if #results == 0 then
        quarto.log.warning("Found no matches for object:\n", search_object)
    end

    return nil
end

local function mysplit (inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={}
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            table.insert(t, str)
    end
    return t
end

local function normalize_role(role)
    if role == "func" then
        return "function"
    end
    return role
end

local function build_search_object(str)
    local starts_with_colon = str:sub(1, 1) == ":"
    local search = {}
    if starts_with_colon then
        local t = mysplit(str, ":")
        if #t == 2 then
            -- e.g. :py:func:`my_func`
            search.role = normalize_role(t[1])
            search.name = t[2]:match("%%60(.*)%%60")
        elseif #t == 3 then
            -- e.g. :py:func:`my_func`
            search.domain = t[1]
            search.role = normalize_role(t[2])
            search.name = t[3]:match("%%60(.*)%%60")
        elseif #t == 4 then
            -- e.g. :ext+inv:py:func:`my_func`
            search.external = true

            search.inv_name = t[1]:match("external%+(.*)")
            search.domain = t[2]
            search.role = normalize_role(t[3])
            search.name = t[4]:match("%%60(.*)%%60")
        else
            quarto.log.warning("couldn't parse this link: " .. str)
            return {}
        end
    else
        search.name = str:match("%%60(.*)%%60")
    end

    if search.name == nil then
        quarto.log.warning("couldn't parse this link: " .. str)
        return {}
    end

    if search.name:sub(1, 1) == "~" then
        search.shortened = true
        search.name = search.name:sub(2, -1)
    end
    return search
end

local function report_broken_link(link, search_object, replacement)
    -- TODO: how to unescape html elements like [?
    return pandoc.Code(pandoc.utils.stringify(link.content))
end

function Link(link)
    -- do not process regular links ----
    if not link.target:match("%%60") then
        return link
    end

    -- lookup item ----
    local search = build_search_object(link.target)
    local item = lookup(search)

    -- determine replacement, used if no link text specified ----
    local original_text = pandoc.utils.stringify(link.content)
    local replacement = search.name
    if search.shortened then
        local t = mysplit(search.name, ".")
        replacement = t[#t]
    end

    -- set link text ----
    if original_text == "" and replacement ~= nil then
        link.content = pandoc.Code(replacement)
    end

    -- report broken links ----
    if item == nil then
        return report_broken_link(link, search)
    end
    link.target = item.uri:gsub("%$$", search.name)


    return link
end

local function fixup_json(json, prefix)
    for _, item in ipairs(json.items) do
        item.uri = prefix .. item.uri
    end
    table.insert(inventory, json)
end

return {
    {
        Meta = function(meta)
            local json
            local prefix
            for k, v in pairs(meta.interlinks.sources) do
                local base_name = quarto.project.offset .. "/_inv/" .. k .. "_objects"
                json = read_inv_text_or_json(base_name)
                prefix = pandoc.utils.stringify(v.url)
                if json ~= nil then
                    fixup_json(json, prefix)
                end
            end
            json = read_inv_text_or_json(quarto.project.offset .. "/objects")
            if json ~= nil then
                fixup_json(json, "/")
            end
        end
    },
    {
        Link = Link
    }
}
