import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Category, Client } from "../../../components/types";
import { MemoryTable } from "./MemoryTable";
import { MemoryPagination } from "./MemoryPagination";
import { CreateMemoryDialog } from "./CreateMemoryDialog";
import { PageSizeSelector } from "./PageSizeSelector";
import { useMemoriesApi } from "@/hooks/useMemoriesApi";
import { useRouter, useSearchParams } from "next/navigation";
import { MemoryTableSkeleton } from "@/skeleton/MemoryTableSkeleton";
import { RefreshCw, Activity } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

export function MemoriesSection() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { fetchMemories } = useMemoriesApi();
  const { toast } = useToast();
  const [memories, setMemories] = useState<any[]>([]);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const currentPage = Number(searchParams.get("page")) || 1;
  const itemsPerPage = Number(searchParams.get("size")) || 10;
  const [selectedCategory, setSelectedCategory] = useState<Category | "all">(
    "all"
  );
  const [selectedClient, setSelectedClient] = useState<Client | "all">("all");

  // 记忆数据加载函数
  const loadMemories = useCallback(async (showRefreshing = false) => {
    if (showRefreshing) {
      setIsRefreshing(true);
    } else {
      setIsLoading(true);
    }
    
    try {
      const searchQuery = searchParams.get("search") || "";
      const result = await fetchMemories(
        searchQuery,
        currentPage,
        itemsPerPage
      );
      setMemories(result.memories);
      setTotalItems(result.total);
      setTotalPages(result.pages);
      setLastUpdated(new Date());
      
      if (showRefreshing) {
        toast({
          title: "数据已更新",
          description: "记忆列表已刷新",
        });
      }
    } catch (error) {
      console.error("Failed to fetch memories:", error);
      toast({
        title: "加载失败",
        description: "无法获取记忆数据，请稍后重试",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, [searchParams, currentPage, itemsPerPage, fetchMemories, toast]);

  // 手动刷新函数
  const handleManualRefresh = useCallback(() => {
    loadMemories(true);
  }, [loadMemories]);

  // 切换自动刷新
  const toggleAutoRefresh = useCallback(() => {
    setAutoRefresh(prev => !prev);
  }, []);

  // 自动刷新逻辑
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(() => {
        loadMemories(false);
      }, 30000); // 30秒自动刷新一次
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, loadMemories]);

  // 初始加载和参数变化时加载
  useEffect(() => {
    loadMemories();
  }, [currentPage, itemsPerPage, searchParams]);

  const setCurrentPage = (page: number) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", page.toString());
    params.set("size", itemsPerPage.toString());
    router.push(`?${params.toString()}`);
  };

  const handlePageSizeChange = (size: number) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", "1"); // Reset to page 1 when changing page size
    params.set("size", size.toString());
    router.push(`?${params.toString()}`);
  };

  if (isLoading) {
    return (
      <div className="w-full bg-transparent">
        <MemoryTableSkeleton />
        <div className="flex items-center justify-between mt-4">
          <div className="h-8 w-32 bg-zinc-800 rounded animate-pulse" />
          <div className="h-8 w-48 bg-zinc-800 rounded animate-pulse" />
          <div className="h-8 w-32 bg-zinc-800 rounded animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div className="w-full bg-transparent">
      {/* 刷新控制栏 */}
      <div className="flex items-center justify-between mb-4 p-3 bg-zinc-900/50 rounded-lg border border-zinc-800">
        <div className="flex items-center gap-3">
          <div className="text-sm text-zinc-400">
            最后更新: {lastUpdated.toLocaleTimeString('zh-CN')}
          </div>
          <div className="flex items-center gap-1">
            <Activity className={`h-4 w-4 ${autoRefresh ? 'text-green-500' : 'text-zinc-500'}`} />
            <span className="text-sm text-zinc-400">
              {autoRefresh ? '自动刷新已启用' : '自动刷新已停用'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={toggleAutoRefresh}
          >
            <Activity className={`h-4 w-4 mr-2 ${autoRefresh ? 'text-green-500' : 'text-zinc-500'}`} />
            {autoRefresh ? '停用自动刷新' : '启用自动刷新'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleManualRefresh}
            disabled={isLoading || isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            {isRefreshing ? '刷新中...' : '手动刷新'}
          </Button>
        </div>
      </div>

      <div>
        {memories.length > 0 ? (
          <>
            <MemoryTable />
            <div className="flex items-center justify-between mt-4">
              <PageSizeSelector
                pageSize={itemsPerPage}
                onPageSizeChange={handlePageSizeChange}
              />
              <div className="text-sm text-zinc-500 mr-2">
                Showing {(currentPage - 1) * itemsPerPage + 1} to{" "}
                {Math.min(currentPage * itemsPerPage, totalItems)} of{" "}
                {totalItems} memories
              </div>
              <MemoryPagination
                currentPage={currentPage}
                totalPages={totalPages}
                setCurrentPage={setCurrentPage}
              />
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="rounded-full bg-zinc-800 p-3 mb-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-6 w-6 text-zinc-400"
              >
                <path d="M21 9v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path>
                <path d="M16 2v6h6"></path>
                <path d="M12 18v-6"></path>
                <path d="M9 15h6"></path>
              </svg>
            </div>
            <h3 className="text-lg font-medium">No memories found</h3>
            <p className="text-zinc-400 mt-1 mb-4">
              {selectedCategory !== "all" || selectedClient !== "all"
                ? "Try adjusting your filters"
                : "Create your first memory to see it here"}
            </p>
            {selectedCategory !== "all" || selectedClient !== "all" ? (
              <Button
                variant="outline"
                onClick={() => {
                  setSelectedCategory("all");
                  setSelectedClient("all");
                }}
              >
                Clear Filters
              </Button>
            ) : (
              <CreateMemoryDialog />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
