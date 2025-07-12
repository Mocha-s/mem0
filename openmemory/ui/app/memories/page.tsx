"use client";

import { useEffect } from "react";
import { MemoriesSection } from "@/app/memories/components/MemoriesSection";
import { MemoryFilters } from "@/app/memories/components/MemoryFilters";
import { SystemMemoriesSection } from "@/components/system/SystemMemoriesSection";
import { DebugInfo } from "@/components/debug/DebugInfo";
import { APITestPage } from "@/components/test/APITestPage";
import { useRouter, useSearchParams } from "next/navigation";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import "@/styles/animation.css";
import UpdateMemory from "@/components/shared/update-memory";
import { useUI } from "@/hooks/useUI";

export default function MemoriesPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { updateMemoryDialog, handleCloseUpdateMemoryDialog } = useUI();
  
  useEffect(() => {
    // Set default pagination values if not present in URL
    if (!searchParams.has("page") || !searchParams.has("size")) {
      const params = new URLSearchParams(searchParams.toString());
      if (!searchParams.has("page")) params.set("page", "1");
      if (!searchParams.has("size")) params.set("size", "10");
      router.push(`?${params.toString()}`);
    }
  }, []);

  return (
    <div className="">
      <UpdateMemory
        memoryId={updateMemoryDialog.memoryId || ""}
        memoryContent={updateMemoryDialog.memoryContent || ""}
        open={updateMemoryDialog.isOpen}
        onOpenChange={handleCloseUpdateMemoryDialog}
      />
      <main className="flex-1 py-6">
        <div className="container">
          <div className="animate-fade-slide-down">
            <Tabs defaultValue="user-memories" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="user-memories">我的记忆</TabsTrigger>
                <TabsTrigger value="system-memories">系统记忆</TabsTrigger>
                <TabsTrigger value="debug">调试信息</TabsTrigger>
                <TabsTrigger value="api-test">API测试</TabsTrigger>
              </TabsList>
              
              <TabsContent value="user-memories" className="space-y-4">
                <div className="mt-1 pb-4 animate-fade-slide-down">
                  <MemoryFilters />
                </div>
                <div className="animate-fade-slide-down delay-1">
                  <MemoriesSection />
                </div>
              </TabsContent>
              
              <TabsContent value="system-memories" className="space-y-4">
                <div className="animate-fade-slide-down delay-1">
                  <SystemMemoriesSection />
                </div>
              </TabsContent>

              <TabsContent value="debug" className="space-y-4">
                <div className="animate-fade-slide-down delay-1">
                  <DebugInfo />
                </div>
              </TabsContent>

              <TabsContent value="api-test" className="space-y-4">
                <div className="animate-fade-slide-down delay-1">
                  <APITestPage />
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  );
}
